-- ==========================================================================
-- Terrain Modification Prototype - Super Mario Bros 1-1
-- ==========================================================================
-- Experimental: intercepts PPU nametable data per-frame to modify terrain.
-- Replaces pipe tiles with sky, or patches ground to create/fill gaps.
--
-- WARNING: This is a prototype. SMB overwrites nametable columns on scroll,
-- so modifications only persist on the currently visible screens. The game's
-- collision detection uses separate metatile data in CPU RAM ($0500 area),
-- so visual changes may not affect collision without patching that too.
--
-- Usage: Load ROM in FCEUX, create save state in slot 1, load this script.
-- ==========================================================================

-- ========================
-- SMB TILE ID REFERENCE
-- ========================
-- These are the pattern table indices SMB uses for various tiles.
-- Determined by examining PPU nametable dumps during gameplay.

local TILE = {
    SKY          = 0x24,  -- empty sky background
    GROUND_TOP_L = 0xB4,  -- ground surface left half
    GROUND_TOP_R = 0xB5,  -- ground surface right half
    GROUND_MID_L = 0xB6,  -- ground body left half
    GROUND_MID_R = 0xB7,  -- ground body right half
    -- Pipe tiles (each pipe column is 2 tiles wide, variable height)
    PIPE_TOP_L   = 0x70,  -- pipe cap left
    PIPE_TOP_R   = 0x71,  -- pipe cap right
    PIPE_BODY_L  = 0x72,  -- pipe shaft left
    PIPE_BODY_R  = 0x73,  -- pipe shaft right
    -- Brick/block tiles
    BRICK        = 0x45,  -- breakable brick
    QBLOCK       = 0xC0,  -- question block (active)
    QBLOCK_DEAD  = 0xC4,  -- question block (hit)
    SOLID_BLOCK  = 0x61,  -- solid block (unbreakable)
}

-- PPU nametable addresses (NES uses $2000-$2FFF for nametables)
-- SMB with horizontal mirroring: NT0=$2000, NT1=$2400
local PPU_NT0 = 0x2000  -- left screen nametable
local PPU_NT1 = 0x2400  -- right screen nametable
local NT_COLS = 32       -- tiles per row
local NT_ROWS = 30       -- rows per nametable

-- ========================
-- HELPERS
-- ========================

local function ppu_nt_addr(ntable, col, row)
    local base = (ntable == 0) and PPU_NT0 or PPU_NT1
    return base + row * NT_COLS + col
end

local function read_ppu_tile(ntable, col, row)
    return ppu.readbyte(ppu_nt_addr(ntable, col, row))
end

local function write_ppu_tile(ntable, col, row, tile_id)
    ppu.writebyte(ppu_nt_addr(ntable, col, row), tile_id)
end

local function is_pipe_tile(tile_id)
    return tile_id == TILE.PIPE_TOP_L or tile_id == TILE.PIPE_TOP_R
        or tile_id == TILE.PIPE_BODY_L or tile_id == TILE.PIPE_BODY_R
end

local function is_ground_tile(tile_id)
    return tile_id == TILE.GROUND_TOP_L or tile_id == TILE.GROUND_TOP_R
        or tile_id == TILE.GROUND_MID_L or tile_id == TILE.GROUND_MID_R
end

-- ========================
-- TERRAIN MODIFICATION MODES
-- ========================

-- Mode: Remove all pipes (replace with sky)
local function remove_pipes_on_screen()
    local changes = 0
    for nt = 0, 1 do
        for col = 0, NT_COLS - 1 do
            for row = 0, NT_ROWS - 1 do
                local tile = read_ppu_tile(nt, col, row)
                if is_pipe_tile(tile) then
                    write_ppu_tile(nt, col, row, TILE.SKY)
                    changes = changes + 1
                end
            end
        end
    end
    return changes
end

-- Mode: Remove ground tiles in specific columns (create gaps)
-- gap_cols: list of {ntable, col_start, col_end} to clear ground from
local function create_gaps(gap_cols)
    local changes = 0
    for _, gap in ipairs(gap_cols) do
        local nt = gap[1]
        for col = gap[2], gap[3] do
            for row = 0, NT_ROWS - 1 do
                local tile = read_ppu_tile(nt, col, row)
                if is_ground_tile(tile) then
                    write_ppu_tile(nt, col, row, TILE.SKY)
                    changes = changes + 1
                end
            end
        end
    end
    return changes
end

-- Mode: Fill sky with ground in specific columns (fill gaps)
local function fill_gaps(fill_cols)
    local changes = 0
    for _, fill in ipairs(fill_cols) do
        local nt = fill[1]
        -- Ground is in bottom 2 rows of the nametable (rows 26-29)
        for col = fill[2], fill[3] do
            -- Row 26-27: ground surface
            write_ppu_tile(nt, col, 26, TILE.GROUND_TOP_L)
            write_ppu_tile(nt, col, 27, TILE.GROUND_MID_L)
            -- Row 28-29: if visible
            if NT_ROWS > 28 then
                write_ppu_tile(nt, col, 28, TILE.GROUND_MID_L)
            end
            if NT_ROWS > 29 then
                write_ppu_tile(nt, col, 29, TILE.GROUND_MID_L)
            end
            changes = changes + 4
        end
    end
    return changes
end

-- ========================
-- DIAGNOSTIC: DUMP VISIBLE TILES
-- ========================

local function dump_nametable_row(nt, row)
    local line = string.format("NT%d Row%02d: ", nt, row)
    for col = 0, 31 do
        local tile = read_ppu_tile(nt, col, row)
        if tile ~= TILE.SKY and tile ~= 0x00 then
            line = line .. string.format("%02X", tile)
        else
            line = line .. ".."
        end
        if col % 4 == 3 then line = line .. " " end
    end
    return line
end

local function dump_nametable_summary()
    print("=== PPU Nametable Tile Dump ===")
    -- Only dump rows 20-29 (ground area where interesting tiles are)
    for nt = 0, 1 do
        print(string.format("--- Nametable %d ---", nt))
        for row = 20, 29 do
            print(dump_nametable_row(nt, row))
        end
    end
end

-- ========================
-- OVERLAY
-- ========================

local function draw_terrain_overlay(mode, pipe_count, ground_changes)
    gui.drawbox(0, 0, 255, 30, "#000000C0", "#000000C0")
    gui.text(2, 2,  "TERRAIN MOD PROTOTYPE", "#FF4444")
    gui.text(2, 12, string.format("Mode: %s  Pipes removed: %d  Ground: %d",
        mode, pipe_count, ground_changes), "#FFFFFF")
    gui.text(2, 22, "NOTE: visual only -- collision uses separate metatile RAM",
        "#FFFF44")
end

-- ========================
-- MAIN LOOP
-- ========================

print("=== Terrain Modification Prototype ===")
print("This modifies PPU nametable tiles per-frame.")
print("Modes: 'no_pipes', 'dump', 'none'")
print("")

-- Configure mode here:
local MODE = "no_pipes"  -- "no_pipes", "dump", "dump_late", "none"

-- Load save state
savestate.load(savestate.object(1))
for i = 1, 30 do emu.frameadvance() end

-- Initial diagnostic dump
if MODE == "dump" then
    dump_nametable_summary()
end

print("Starting terrain modification loop (mode=" .. MODE .. ")")

emu.speedmode("normal")
local frame = 0
local total_pipe_changes = 0
local total_ground_changes = 0

-- In dump_late mode, hold right to scroll Mario to the first pipe, then dump
while true do
    frame = frame + 1
    
    local pipe_changes = 0
    local ground_changes = 0
    
    if MODE == "no_pipes" then
        -- Remove pipes every frame (they get rewritten on scroll)
        pipe_changes = remove_pipes_on_screen()
        if pipe_changes > 0 and total_pipe_changes == 0 then
            print(string.format("  First pipe tiles found at frame %d!", frame))
        end
        total_pipe_changes = total_pipe_changes + pipe_changes
    end
    
    -- Dump at startup
    if MODE == "dump" and frame == 60 then
        dump_nametable_summary()
    end
    
    -- Hold right+B to scroll Mario, dump when near first pipe (~400 frames)
    if MODE == "dump_late" then
        joypad.set(1, {right = true, B = true})
        if frame == 350 or frame == 400 or frame == 450 then
            print(string.format("\n=== DUMP AT FRAME %d (near pipes) ===", frame))
            -- Dump ALL rows, not just 20-29
            for nt = 0, 1 do
                print(string.format("--- Nametable %d ---", nt))
                for row = 0, 29 do
                    print(dump_nametable_row(nt, row))
                end
            end
        end
    end
    
    draw_terrain_overlay(MODE, pipe_changes, ground_changes)
    emu.frameadvance()
end
