-- ==========================================================================
-- Level Randomizer - Super Mario Bros World 1-1
-- ==========================================================================
-- Standalone FCEUX Lua script that randomizes enemy positions, types, and
-- timer values in NES RAM after a save state load. Creates level variants
-- that force neural networks (NEAT or DQN) to generalize instead of
-- memorizing the specific World 1-1 layout.
--
-- Usage:
--   Standalone: Load ROM in FCEUX, create save state in slot 1 at 1-1
--               start, then load this script. It will demo randomized
--               variants in a loop.
--
--   From another script: dofile("level_randomizer.lua")
--                         randomize_level("combined")
--
-- Modes: "enemies", "timer", "combined"
-- ==========================================================================

-- ========================
-- CONFIGURATION
-- ========================

local RANDOMIZER_CONFIG = {
    SAVESTATE_SLOT   = 1,
    SETTLE_FRAMES    = 30,     -- frames to let NES settle after state load
    DEMO_PLAY_FRAMES = 900,    -- frames to show each demo run (~15 seconds at 60fps)
    DEMO_LOOP_DELAY  = 120,    -- pause frames between demo loops (2 seconds)

    -- Enemy randomization parameters
    ENEMY_X_SHIFT_MIN  = -2,   -- tiles (each tile = 16px on screen)
    ENEMY_X_SHIFT_MAX  =  2,
    SWAP_TYPE_CHANCE    = 0.30, -- 30% chance to swap Goomba <-> Koopa
    REMOVE_ENEMY_CHANCE = 0.20, -- 20% chance to remove each enemy
    ADD_ENEMY_CHANCE    = 0.10, -- 10% chance to add an extra Goomba

    -- Timer options (BCD digit values, not raw numbers)
    TIMER_OPTIONS = {
        { h = 3, t = 5, o = 0 },  -- 350
        { h = 3, t = 7, o = 5 },  -- 375
        { h = 4, t = 0, o = 0 },  -- 400 (original)
    },
}

-- ========================
-- MEMORY ADDRESSES
-- ========================
-- Matches neat_evolve.lua / genetic_bruteforce.lua / mario_ai.lua

local ADDR = {
    -- Mario
    MARIO_X_PAGE  = 0x006D,
    MARIO_X_POS   = 0x0086,
    MARIO_Y       = 0x00CE,
    PLAYER_STATE  = 0x000E,
    FLOAT_STATE   = 0x001D,

    -- Timer (BCD digits)
    TIME_H        = 0x07F8,
    TIME_M        = 0x07F9,
    TIME_L        = 0x07FA,

    -- Lives
    LIVES         = 0x075A,

    -- Enemy slots (5 enemies max on screen at once)
    ENEMY_TYPE    = { 0x0016, 0x0017, 0x0018, 0x0019, 0x001A },
    ENEMY_X       = { 0x0087, 0x0088, 0x0089, 0x008A, 0x008B },
    ENEMY_Y       = { 0x00CF, 0x00D0, 0x00D1, 0x00D2, 0x00D3 },
    ENEMY_X_PAGE  = { 0x006E, 0x006F, 0x0070, 0x0071, 0x0072 },
}

-- Enemy type constants
local ENEMY = {
    KOOPA_GREEN = 0x00,
    KOOPA_RED   = 0x02,
    GOOMBA      = 0x06,
    EMPTY       = 0xFF,
}

-- Valid ground Y positions for spawned enemies (screen-relative)
-- In SMB 1-1, ground level is at Y ~192; platform enemies sit at ~144, ~160
local VALID_ENEMY_Y = { 0xA8, 0xB0, 0xB8, 0xC0 }  -- 168, 176, 184, 192

-- ========================
-- HELPER FUNCTIONS
-- ========================

local function get_mario_x()
    return memory.readbyte(ADDR.MARIO_X_PAGE) * 256
         + memory.readbyte(ADDR.MARIO_X_POS)
end

local function get_timer()
    return memory.readbyte(ADDR.TIME_H) * 100
         + memory.readbyte(ADDR.TIME_M) * 10
         + memory.readbyte(ADDR.TIME_L)
end

local function is_dead()
    local state = memory.readbyte(ADDR.PLAYER_STATE)
    return state == 0x0B or state == 0x06
end

--- Clamp a value to [lo, hi].
local function clamp(val, lo, hi)
    if val < lo then return lo end
    if val > hi then return hi end
    return val
end

--- Return true if the given enemy type byte is a real enemy (not empty/unused).
local function is_active_enemy(etype)
    return etype ~= ENEMY.EMPTY and etype ~= 0xFF
end

--- Return true if the given type is Goomba or Koopa (swappable).
local function is_swappable(etype)
    return etype == ENEMY.GOOMBA
        or etype == ENEMY.KOOPA_GREEN
        or etype == ENEMY.KOOPA_RED
end

-- ========================
-- CHANGE LOG (for overlay)
-- ========================

local last_changes = {}  -- list of human-readable change descriptions

local function log_change(msg)
    last_changes[#last_changes + 1] = msg
end

local function clear_changes()
    last_changes = {}
end

-- ========================
-- ENEMY RANDOMIZATION
-- ========================

--- Shuffle enemy X positions, swap types, remove/add enemies.
--- Operates on the 5 enemy slots currently loaded in NES RAM.
local function shuffle_enemies()
    local cfg = RANDOMIZER_CONFIG

    -- -- Phase 1: modify existing enemy slots ----------------------------------
    for slot = 1, 5 do
        local etype = memory.readbyte(ADDR.ENEMY_TYPE[slot])

        -- Only process active enemy slots
        if is_active_enemy(etype) then
            local removed = false

            -- (a) Random removal
            if math.random() < cfg.REMOVE_ENEMY_CHANCE then
                memory.writebyte(ADDR.ENEMY_TYPE[slot], ENEMY.EMPTY)
                log_change(string.format("Slot %d: removed (was 0x%02X)", slot, etype))
                removed = true
            end

            if not removed then
                -- (b) Type swap (Goomba <-> Koopa)
                if is_swappable(etype) and math.random() < cfg.SWAP_TYPE_CHANCE then
                    local new_type
                    if etype == ENEMY.GOOMBA then
                        -- 50/50 green or red Koopa
                        new_type = (math.random() < 0.5) and ENEMY.KOOPA_GREEN
                                                           or ENEMY.KOOPA_RED
                    else
                        new_type = ENEMY.GOOMBA
                    end
                    memory.writebyte(ADDR.ENEMY_TYPE[slot], new_type)
                    log_change(string.format("Slot %d: type 0x%02X -> 0x%02X",
                                             slot, etype, new_type))
                end

                -- (c) X position shift (-2..+2 tiles, 1 tile = 16px)
                local shift_tiles = math.random(cfg.ENEMY_X_SHIFT_MIN,
                                                cfg.ENEMY_X_SHIFT_MAX)
                if shift_tiles ~= 0 then
                    local old_x = memory.readbyte(ADDR.ENEMY_X[slot])
                    local new_x = clamp(old_x + shift_tiles * 16, 0, 255)
                    memory.writebyte(ADDR.ENEMY_X[slot], new_x)
                    log_change(string.format("Slot %d: X %d -> %d (%+d tiles)",
                                             slot, old_x, new_x, shift_tiles))
                end
            end
        end  -- is_active_enemy
    end

    -- -- Phase 2: maybe add an extra Goomba in the first empty slot -----------
    if math.random() < cfg.ADD_ENEMY_CHANCE then
        for slot = 1, 5 do
            local etype = memory.readbyte(ADDR.ENEMY_TYPE[slot])
            if not is_active_enemy(etype) then
                -- Place a Goomba at a random valid ground position
                local mario_page = memory.readbyte(ADDR.MARIO_X_PAGE)
                local spawn_x    = math.random(80, 240)
                local spawn_y    = VALID_ENEMY_Y[math.random(#VALID_ENEMY_Y)]

                memory.writebyte(ADDR.ENEMY_TYPE[slot],   ENEMY.GOOMBA)
                memory.writebyte(ADDR.ENEMY_X[slot],      spawn_x)
                memory.writebyte(ADDR.ENEMY_Y[slot],      spawn_y)
                memory.writebyte(ADDR.ENEMY_X_PAGE[slot], mario_page)

                log_change(string.format(
                    "Slot %d: +Goomba at X=%d Y=0x%02X page=%d",
                    slot, spawn_x, spawn_y, mario_page))
                break  -- only add one
            end
        end
    end
end

-- ========================
-- TIMER RANDOMIZATION
-- ========================

--- Set the in-game timer to a randomly chosen value from TIMER_OPTIONS.
local function vary_timer()
    local opts  = RANDOMIZER_CONFIG.TIMER_OPTIONS
    local pick  = opts[math.random(#opts)]
    local old_t = get_timer()

    memory.writebyte(ADDR.TIME_H, pick.h)
    memory.writebyte(ADDR.TIME_M, pick.t)
    memory.writebyte(ADDR.TIME_L, pick.o)

    local new_t = pick.h * 100 + pick.t * 10 + pick.o
    if new_t ~= old_t then
        log_change(string.format("Timer: %d -> %d", old_t, new_t))
    end
end

-- ========================
-- MAIN PUBLIC API
-- ========================

-- Per-frame Goomba->Koopa conversion (runs every frame when active)
local koopa_world_active = false

local function koopa_world_tick()
    if not koopa_world_active then return end
    for slot = 1, 5 do
        local etype = memory.readbyte(ADDR.ENEMY_TYPE[slot])
        if etype == ENEMY.GOOMBA then
            local new_type = (math.random() < 0.5) and ENEMY.KOOPA_GREEN or ENEMY.KOOPA_RED
            memory.writebyte(ADDR.ENEMY_TYPE[slot], new_type)
        end
    end
end

--- Enable/disable continuous Goomba->Koopa conversion (runs every frame).
--- Unlike shuffle_enemies() which is one-shot, this catches newly spawned enemies.
function set_koopa_world(enabled)
    koopa_world_active = (enabled ~= false)
    if koopa_world_active then
        log_change("Koopa World: ALL Goombas will become Koopas (per-frame)")
    end
end

--- Randomize the current level in NES RAM.
--- @param mode string  "enemies" | "timer" | "combined" | "koopa_world" (default "combined")
--- Loads save state, waits for settle, then applies randomization.
--- "koopa_world" mode: converts every Goomba to Koopa continuously (per-frame hook).
--- Designed to be called from other scripts via dofile().
function randomize_level(mode)
    mode = mode or "combined"
    clear_changes()

    -- Load save state
    savestate.load(savestate.object(RANDOMIZER_CONFIG.SAVESTATE_SLOT))

    -- Let the NES settle (PPU, scroll registers, etc.)
    for i = 1, RANDOMIZER_CONFIG.SETTLE_FRAMES do
        emu.frameadvance()
    end

    -- Apply randomizations
    if mode == "koopa_world" then
        set_koopa_world(true)
        vary_timer()
    elseif mode == "enemies" or mode == "combined" then
        shuffle_enemies()
    end
    if mode == "timer" or mode == "combined" then
        vary_timer()
    end

    -- Log summary
    if #last_changes > 0 then
        print(string.format("[randomizer] Applied %d changes (mode=%s):",
                            #last_changes, mode))
        for _, msg in ipairs(last_changes) do
            print("  " .. msg)
        end
    else
        print("[randomizer] No changes applied (mode=" .. mode .. ")")
    end
end

--- Return the list of change descriptions from the last randomize_level() call.
--- Useful for overlay rendering in the calling script.
function get_randomizer_changes()
    return last_changes
end

-- ========================
-- OVERLAY (standalone use)
-- ========================

local function draw_overlay(frame_count)
    -- Semi-transparent background panel
    gui.drawbox(0, 0, 255, 52, "#000000B0", "#000000B0")
    gui.text(2, 2, "LEVEL RANDOMIZER  [standalone demo]", "#44FF44")
    gui.text(2, 12,
        string.format("Mario X: %d  Timer: %d  Frame: %d",
                       get_mario_x(), get_timer(), frame_count),
        "#FFFFFF")

    -- Show up to 3 most recent changes
    local y = 22
    local show = math.min(#last_changes, 3)
    for i = 1, show do
        gui.text(2, y, last_changes[i], "#FFFF44")
        y = y + 10
    end
    if #last_changes > 3 then
        gui.text(2, y, string.format("... +%d more", #last_changes - 3), "#AAAAAA")
    end
end

-- ========================
-- STANDALONE DEMO LOOP
-- ========================
-- Only runs when the script is executed directly (not dofile'd).
-- Detection: if randomize_level is the only global we defined AND no
-- caller has set _RANDOMIZER_IMPORTED, we assume standalone mode.

if not _RANDOMIZER_IMPORTED then
    -- Seed RNG
    math.randomseed(os.time())

    print("=== Level Randomizer ===")
    print("Modes: 'combined' (default), 'enemies', 'timer', 'koopa_world'")
    print("Koopa World: every Goomba becomes a Koopa (per-frame, catches new spawns)")
    print("Set  _RANDOMIZER_IMPORTED = true  before dofile() to suppress this.")

    -- Change this to try different modes:
    local MODE = "koopa_world"   -- try "combined", "enemies", "timer", or "koopa_world"

    -- Randomize once, then let the user play with overlay
    randomize_level(MODE)

    emu.speedmode("normal")
    local frame = 0
    while true do
        frame = frame + 1
        koopa_world_tick()  -- converts any Goomba to Koopa if koopa_world is active
        draw_overlay(frame)
        emu.frameadvance()
    end
end
