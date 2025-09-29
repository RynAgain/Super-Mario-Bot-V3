# Super Mario Bros NES Memory Address Reference

## Overview

This document provides comprehensive memory address mappings for Super Mario Bros (NES) to extract all game state information needed for AI training.

## Core Mario State

### Position and Movement
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x006D | 1 | Mario X Position (screen) | 0x00-0xFF |
| 0x0086 | 1 | Mario X Position (level) | 0x00-0xFF |
| 0x03AD | 1 | Mario X Position (level, high byte) | 0x00-0xFF |
| 0x00CE | 1 | Mario Y Position (screen) | 0x00-0xFF |
| 0x03B8 | 1 | Mario Y Position (level) | 0x00-0xFF |
| 0x0057 | 1 | Mario X Velocity | 0x00-0xFF (signed) |
| 0x009F | 1 | Mario Y Velocity | 0x00-0xFF (signed) |
| 0x0045 | 1 | Mario Direction | 0x01=right, 0x02=left |
| 0x001D | 1 | Mario on Ground Flag | 0x00=air, 0x01=ground |

### Power State and Status
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x0756 | 1 | Mario Power State | 0x00=small, 0x01=big, 0x02=fire |
| 0x075A | 1 | Lives Remaining | 0x00-0xFF |
| 0x079E | 1 | Invincibility Timer | 0x00-0xFF |
| 0x0770 | 1 | Star Power Timer | 0x00-0xFF |
| 0x0079 | 1 | Mario Animation State | Various values |
| 0x001E | 1 | Mario Crouching | 0x00=no, 0x08=yes |

## Level and World Information

### Level Progress
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x075F | 1 | World Number | 0x00-0x07 (1-8) |
| 0x0760 | 1 | Level Number | 0x00-0x03 (1-4) |
| 0x03AD | 1 | Level X Position (high) | 0x00-0xFF |
| 0x0086 | 1 | Level X Position (low) | 0x00-0xFF |
| 0x071A | 1 | Level Timer (hundreds) | 0x00-0x09 |
| 0x071B | 1 | Level Timer (tens) | 0x00-0x09 |
| 0x071C | 1 | Level Timer (ones) | 0x00-0x09 |

### Camera and Screen
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x03AD | 1 | Screen X Position (high) | 0x00-0xFF |
| 0x0086 | 1 | Screen X Position (low) | 0x00-0xFF |
| 0x00B5 | 1 | Screen Y Position | 0x00-0xFF |
| 0x0725 | 1 | Vertical Scroll | 0x00-0xFF |

## Score and Collectibles

### Score System
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x07DD | 1 | Score (100,000s) | 0x00-0x09 |
| 0x07DE | 1 | Score (10,000s) | 0x00-0x09 |
| 0x07DF | 1 | Score (1,000s) | 0x00-0x09 |
| 0x07E0 | 1 | Score (100s) | 0x00-0x09 |
| 0x07E1 | 1 | Score (10s) | 0x00-0x09 |
| 0x07E2 | 1 | Score (1s) | 0x00-0x09 |

### Coins and Items
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x075E | 1 | Coins (tens) | 0x00-0x09 |
| 0x075D | 1 | Coins (ones) | 0x00-0x09 |
| 0x0772 | 1 | 1-Up Flag | 0x00=no, 0x01=yes |

## Enemy Information

### Enemy Positions (8 enemy slots)
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x0087-0x008E | 8 | Enemy X Position (screen) | 0x00-0xFF each |
| 0x00CF-0x00D6 | 8 | Enemy Y Position (screen) | 0x00-0xFF each |
| 0x0014-0x001B | 8 | Enemy Type | Various enemy IDs |
| 0x001C-0x0023 | 8 | Enemy State | 0x00=inactive, others=active |

### Enemy Movement and Status
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x0058-0x005F | 8 | Enemy X Velocity | 0x00-0xFF each (signed) |
| 0x00A0-0x00A7 | 8 | Enemy Y Velocity | 0x00-0xFF each (signed) |
| 0x0046-0x004D | 8 | Enemy Direction | 0x01=right, 0x02=left |

## Game State and Control

### Game Status
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x0770 | 1 | Game Engine State | 0x00=playing, others=paused/menu |
| 0x001D | 1 | Player State | Various states |
| 0x0E | 1 | Game Mode | 0x00=demo, 0x01=game, 0x02=ending |
| 0x0772 | 1 | End of Level Flag | 0x00=playing, 0x01=completed |

### Controller Input
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x00F7 | 1 | Controller 1 Input | Bit flags for buttons |
| 0x00F6 | 1 | Controller 1 Previous | Previous frame input |

## Level Layout and Collision

### Block and Tile Information
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x0500-0x069F | 416 | Level Data Buffer | Tile IDs and properties |
| 0x06A0-0x06CF | 48 | Collision Detection Buffer | Collision flags |

### Special Objects
| Address | Size | Description | Range/Values |
|---------|------|-------------|--------------|
| 0x0024-0x002B | 8 | Object Type | Various object IDs |
| 0x008F-0x0096 | 8 | Object X Position | 0x00-0xFF each |
| 0x00D7-0x00DE | 8 | Object Y Position | 0x00-0xFF each |

## Derived Calculations

### Useful Computed Values
These values should be calculated from the raw memory addresses:

#### Mario World Position
```
mario_world_x = (memory[0x03AD] * 256) + memory[0x0086]
mario_world_y = memory[0x03B8]
```

#### Total Score
```
total_score = (memory[0x07DD] * 100000) + 
              (memory[0x07DE] * 10000) + 
              (memory[0x07DF] * 1000) + 
              (memory[0x07E0] * 100) + 
              (memory[0x07E1] * 10) + 
              memory[0x07E2]
```

#### Total Coins
```
total_coins = (memory[0x075E] * 10) + memory[0x075D]
```

#### Time Remaining
```
time_remaining = (memory[0x071A] * 100) + 
                 (memory[0x071B] * 10) + 
                 memory[0x071C]
```

#### Level Progress Percentage
```
# World 1-1 is approximately 3168 pixels long
level_progress = (mario_world_x / 3168.0) * 100
```

## Memory Reading Strategy

### High-Frequency Reads (Every Frame)
- Mario position and velocity
- Enemy positions and states
- Game state flags
- Controller input

### Medium-Frequency Reads (Every 4 frames)
- Score and coins
- Timer
- Power-up states

### Low-Frequency Reads (Every 60 frames)
- World/level numbers
- Lives remaining
- Level layout data

## Implementation Notes

### Endianness
- All NES memory is little-endian
- Multi-byte values should be read accordingly

### BCD Encoding
- Score and timer values use Binary Coded Decimal
- Each nibble represents a decimal digit (0-9)

### Bit Flags
Controller input at 0x00F7:
```
Bit 7: A Button
Bit 6: B Button
Bit 5: Select
Bit 4: Start
Bit 3: Up
Bit 2: Down
Bit 1: Left
Bit 0: Right
```

### Memory Stability
- Some addresses may have temporary invalid values during transitions
- Implement validation checks for critical values
- Use previous frame data if current frame appears corrupted

## Validation Ranges

### Sanity Checks
- Mario X position should be >= previous frame (unless death/reset)
- Mario Y position should be within screen bounds (0-240)
- Enemy positions should be within reasonable screen bounds
- Score should only increase (unless death/reset)
- Timer should only decrease (unless level complete)

This comprehensive memory mapping enables the AI to have complete awareness of the game state for optimal decision making.