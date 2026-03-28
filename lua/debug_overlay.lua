-- ==========================================================================
-- SMB 1-1 Debug Overlay for FCEUX
-- ==========================================================================
-- Standalone Lua script for manual play or observation.
-- Draws game state stats and level feature markers on the FCEUX screen.
-- Does NOT interfere with the AI training system -- run separately.
--
-- Usage: Load in FCEUX via File > Lua > New Lua Script Window
-- ==========================================================================

-- ========================
-- MEMORY ADDRESSES (SMB1)
-- ========================
local ADDR = {
    MARIO_X_PAGE    = 0x006D,   -- X position high byte (screen/page)
    MARIO_X_POS     = 0x0086,   -- X position low byte (within page)
    MARIO_Y         = 0x00CE,   -- Y position on screen
    MARIO_STATE     = 0x000E,   -- Player state (0=leftmost, 1=normal, etc.)
    MARIO_FLOAT     = 0x001D,   -- Float state: 0=ground, 1=air, 2=water
    MARIO_VEL_X     = 0x0057,   -- X velocity (signed)
    MARIO_VEL_Y     = 0x009F,   -- Y velocity (signed)
    MARIO_POWER     = 0x0756,   -- Power state: 0=small, 1=big, 2=fire
    MARIO_DIRECTION = 0x0045,   -- Facing: 1=right, 2=left
    LIVES           = 0x075A,   -- Lives (displayed - 1)
    WORLD           = 0x075F,   -- World (0-indexed)
    LEVEL           = 0x075C,   -- Level (0-indexed)
    SCORE_H         = 0x07DD,   -- Score digits (6 digits, 07DD-07E2)
    COINS           = 0x075E,   -- Coin count
    TIME_H          = 0x07F8,   -- Timer hundreds
    TIME_T          = 0x07F9,   -- Timer tens
    TIME_O          = 0x07FA,   -- Timer ones
    CAMERA_X        = 0x03AD,   -- Camera scroll position (page)
    CAMERA_X_POS    = 0x071C,   -- Camera scroll position (sub)
    ENEMY_DRAWN     = 0x000F,   -- Enemy drawn flag array base
    ENEMY_TYPE      = 0x0016,   -- Enemy type array base (5 slots)
    ENEMY_X_PAGE    = 0x006E,   -- Enemy X page array base
    ENEMY_X_POS     = 0x0087,   -- Enemy X position array base
    ENEMY_Y_POS     = 0x00CF,   -- Enemy Y position array base
}

-- ========================
-- WORLD 1-1 FEATURES MAP
-- ========================
-- Empirically determined X positions of major features in World 1-1.
-- These are absolute X positions (page * 256 + pos).
local FEATURES_1_1 = {
    -- Enemies (approximate spawn X positions)
    {x=352,  type="enemy",   name="Goomba 1"},
    {x=512,  type="enemy",   name="Goomba 2"},
    {x=640,  type="enemy",   name="Goomba 3"},
    {x=816,  type="enemy",   name="Koopa 1"},
    {x=1280, type="enemy",   name="Goomba Pair"},
    {x=1536, type="enemy",   name="Goomba 4"},
    {x=1776, type="enemy",   name="Koopa 2"},
    {x=1840, type="enemy",   name="Goomba 5-6"},
    {x=2064, type="enemy",   name="Goomba 7-8"},
    {x=2560, type="enemy",   name="Goomba 9-10"},
    {x=2688, type="enemy",   name="Goombas 11-12"},
    
    -- Pipes
    {x=448,  type="pipe",    name="Pipe 1 (small)"},
    {x=608,  type="pipe",    name="Pipe 2 (tall)"},
    {x=736,  type="pipe",    name="Pipe 3 (tallest)"},
    {x=912,  type="pipe",    name="Pipe 4 (tall)"},
    {x=2608, type="pipe",    name="Warp Pipe"},
    
    -- Pits (gaps in the ground) -- TODO: verify exact X with manual play
    -- Previous values (430, 880, 1540, 1620) were death distribution clusters,
    -- NOT actual ground gaps. Those deaths are from enemies/pipes.
    -- Approximate real pit locations (need verification with X readout):
    {x=1070, type="pit",     name="Pit 1", x_end=1120},
    {x=1360, type="pit",     name="Pit 2", x_end=1410},
    {x=1430, type="pit",     name="Pit 3", x_end=1470},
    {x=2480, type="pit",     name="Pit 4", x_end=2530},
    
    -- Question blocks and power-ups
    {x=256,  type="block",   name="? Block (Coin)"},
    {x=320,  type="block",   name="? Block (Mushroom)"},
    {x=368,  type="block",   name="? Blocks x3"},
    {x=1248, type="block",   name="? Block (Star)"},
    {x=1504, type="block",   name="? Block (1-Up)"},
    
    -- Stairs/platforms
    {x=2144, type="stairs",  name="Staircase 1"},
    {x=2240, type="stairs",  name="Staircase 2"},
    {x=2480, type="stairs",  name="Staircase 3"},
    {x=2864, type="stairs",  name="Final Staircase"},
    
    -- Landmarks
    {x=40,   type="info",    name="START"},
    {x=3168, type="info",    name="FLAGPOLE"},
}

-- ========================
-- COLORS
-- ========================
local COLORS = {
    enemy  = "#FF4444",
    pipe   = "#44AA44",
    pit    = "#FF8800",
    block  = "#FFFF44",
    stairs = "#8888FF",
    info   = "#FFFFFF",
    bg     = "#000000C0",   -- semi-transparent black
    text   = "#FFFFFF",
    stat   = "#44FF44",
    warn   = "#FF4444",
}

-- ========================
-- HELPER FUNCTIONS
-- ========================

local function get_mario_x()
    local page = memory.readbyte(ADDR.MARIO_X_PAGE)
    local pos = memory.readbyte(ADDR.MARIO_X_POS)
    return page * 256 + pos
end

local function get_timer()
    local h = memory.readbyte(ADDR.TIME_H)
    local t = memory.readbyte(ADDR.TIME_T)
    local o = memory.readbyte(ADDR.TIME_O)
    return h * 100 + t * 10 + o
end

local function get_score()
    local score = 0
    for i = 0, 5 do
        score = score * 10 + memory.readbyte(ADDR.SCORE_H + i)
    end
    return score
end

local function signed_byte(val)
    if val > 127 then return val - 256 end
    return val
end

-- ========================
-- TRACKING STATE
-- ========================
local state = {
    max_x = 0,
    frame_count = 0,
    episode_start_x = 0,
    deaths = 0,
    prev_lives = -1,
    pits_cleared = {},
    features_seen = {},
}

-- ========================
-- MAIN OVERLAY DRAWING
-- ========================

local function draw_overlay()
    state.frame_count = state.frame_count + 1
    
    -- Read game state
    local mario_x = get_mario_x()
    local mario_y = memory.readbyte(ADDR.MARIO_Y)
    local vel_x = signed_byte(memory.readbyte(ADDR.MARIO_VEL_X))
    local vel_y = signed_byte(memory.readbyte(ADDR.MARIO_VEL_Y))
    local float_state = memory.readbyte(ADDR.MARIO_FLOAT)
    local power = memory.readbyte(ADDR.MARIO_POWER)
    local direction = memory.readbyte(ADDR.MARIO_DIRECTION)
    local lives = memory.readbyte(ADDR.LIVES)
    local world = memory.readbyte(ADDR.WORLD) + 1
    local level = memory.readbyte(ADDR.LEVEL) + 1
    local timer = get_timer()
    local score = get_score()
    local coins = memory.readbyte(ADDR.COINS)
    
    -- Track max distance
    if mario_x > state.max_x then
        state.max_x = mario_x
    end
    
    -- Track deaths
    if state.prev_lives >= 0 and lives < state.prev_lives then
        state.deaths = state.deaths + 1
    end
    state.prev_lives = lives
    
    -- Calculate progress
    local progress = math.min(100, mario_x / 3168 * 100)
    local max_progress = math.min(100, state.max_x / 3168 * 100)
    
    -- Float state text
    local float_text = "GND"
    if float_state == 1 then float_text = "AIR"
    elseif float_state == 2 then float_text = "SWIM" end
    
    -- Power state text
    local power_text = "Small"
    if power == 1 then power_text = "Big"
    elseif power == 2 then power_text = "Fire" end
    
    -- Direction text
    local dir_text = direction == 1 and "R" or "L"
    
    -- =====================
    -- TOP-LEFT: Stats Panel
    -- =====================
    gui.drawbox(0, 0, 130, 85, COLORS.bg, COLORS.bg)
    
    gui.text(2, 2,  string.format("X: %d (%.1f%%)", mario_x, progress), COLORS.stat)
    gui.text(2, 12, string.format("MAX: %d (%.1f%%)", state.max_x, max_progress), COLORS.text)
    gui.text(2, 22, string.format("Y: %d  %s  %s", mario_y, float_text, power_text), COLORS.text)
    gui.text(2, 32, string.format("Vel: %d,%d  Dir:%s", vel_x, vel_y, dir_text), COLORS.text)
    gui.text(2, 42, string.format("W%d-%d  T:%d", world, level, timer), COLORS.text)
    gui.text(2, 52, string.format("Score:%d  Coins:%d", score, coins), COLORS.text)
    gui.text(2, 62, string.format("Lives:%d  Deaths:%d", lives + 1, state.deaths), COLORS.text)
    gui.text(2, 72, string.format("Frame: %d", state.frame_count), "#888888")
    
    -- =====================
    -- TOP-RIGHT: Nearest Features
    -- =====================
    gui.drawbox(140, 0, 255, 55, COLORS.bg, COLORS.bg)
    gui.text(142, 2, "-- AHEAD --", "#AAAAAA")
    
    local ahead_count = 0
    for _, feat in ipairs(FEATURES_1_1) do
        if feat.x > mario_x and feat.x < mario_x + 300 and ahead_count < 4 then
            local dist = feat.x - mario_x
            local color = COLORS[feat.type] or COLORS.text
            local name = feat.name
            if #name > 18 then name = string.sub(name, 1, 18) end
            gui.text(142, 12 + ahead_count * 10, string.format("%dpx %s", dist, name), color)
            ahead_count = ahead_count + 1
        end
    end
    
    if ahead_count == 0 then
        gui.text(142, 12, "(clear ahead)", "#666666")
    end
    
    -- =====================
    -- BOTTOM: Pit Warnings (only near ground level to avoid false triggers on pipes/bricks)
    -- =====================
    if mario_y > 140 then
        for _, feat in ipairs(FEATURES_1_1) do
            if feat.type == "pit" then
                local dist = feat.x - mario_x
                if dist > -50 and dist < 80 then
                    local flash = (state.frame_count % 10 < 5) and COLORS.warn or "#FF8800"
                    gui.text(80, 220, string.format("!! PIT: %s [%dpx] !!", feat.name, dist), flash)
                end
            end
        end
    end
    
    -- =====================
    -- MINIMAP BAR (bottom)
    -- =====================
    local bar_y = 230
    local bar_w = 240
    local bar_x = 8
    
    -- Background bar
    gui.drawbox(bar_x, bar_y, bar_x + bar_w, bar_y + 6, "#333333", "#333333")
    
    -- Draw feature markers on minimap
    for _, feat in ipairs(FEATURES_1_1) do
        local px = bar_x + math.floor(feat.x / 3168 * bar_w)
        if px >= bar_x and px <= bar_x + bar_w then
            local color = COLORS[feat.type] or "#888888"
            gui.drawline(px, bar_y, px, bar_y + 6, color)
        end
    end
    
    -- Mario position marker (bright green)
    local mario_px = bar_x + math.floor(mario_x / 3168 * bar_w)
    gui.drawbox(mario_px - 1, bar_y - 1, mario_px + 1, bar_y + 7, "#00FF00", "#00FF00")
    
    -- Max distance marker (dim green)
    local max_px = bar_x + math.floor(state.max_x / 3168 * bar_w)
    gui.drawline(max_px, bar_y - 1, max_px, bar_y + 7, "#008800")
    
    -- =====================
    -- ENEMY TRACKING
    -- =====================
    local enemy_count = 0
    for slot = 0, 4 do
        local drawn = memory.readbyte(ADDR.ENEMY_DRAWN + slot)
        if drawn ~= 0 then
            local etype = memory.readbyte(ADDR.ENEMY_TYPE + slot)
            local ex_page = memory.readbyte(ADDR.ENEMY_X_PAGE + slot)
            local ex_pos = memory.readbyte(ADDR.ENEMY_X_POS + slot)
            local ey = memory.readbyte(ADDR.ENEMY_Y_POS + slot)
            local ex = ex_page * 256 + ex_pos
            enemy_count = enemy_count + 1
            
            -- Screen-relative position for drawing (approximate)
            local camera_page = memory.readbyte(ADDR.CAMERA_X)
            local camera_pos = memory.readbyte(ADDR.CAMERA_X_POS)
            local camera = camera_page * 256 + camera_pos
            local screen_x = ex - camera
            
            if screen_x > 0 and screen_x < 256 then
                -- Draw red marker above enemy
                gui.drawbox(screen_x - 2, ey - 10, screen_x + 2, ey - 6, COLORS.enemy, COLORS.enemy)
                gui.text(screen_x - 4, ey - 18, string.format("E%d", etype), COLORS.enemy)
            end
        end
    end
end

-- ========================
-- REGISTER AND RUN
-- ========================

-- Register the overlay to run every frame after the screen is drawn
gui.register(draw_overlay)

-- Print startup message
print("=== SMB Debug Overlay Loaded ===")
print("Shows: Position, velocity, enemies, features, minimap")
print("Features labeled: enemies, pipes, pits, blocks, stairs")
print("Pit warnings flash when approaching gaps")
print("================================")
