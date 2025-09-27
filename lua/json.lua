--[[
Simple JSON Library for FCEUX Lua
=================================

A lightweight JSON encoder/decoder specifically designed for FCEUX Lua environment.
Supports the basic JSON operations needed for the Mario AI training system.

This implementation focuses on compatibility and simplicity rather than performance
or complete JSON specification compliance.
]]

local json = {}

-- ============================================================================
-- JSON ENCODING
-- ============================================================================

-- Escape special characters in strings
local function escape_string(str)
    local escape_chars = {
        ['"'] = '\\"',
        ['\\'] = '\\\\',
        ['/'] = '\\/',
        ['\b'] = '\\b',
        ['\f'] = '\\f',
        ['\n'] = '\\n',
        ['\r'] = '\\r',
        ['\t'] = '\\t'
    }
    
    return str:gsub('["\\\b\f\n\r\t/]', escape_chars)
end

-- Encode a value to JSON
local function encode_value(value, indent_level)
    indent_level = indent_level or 0
    local value_type = type(value)
    
    if value_type == "nil" then
        return "null"
    elseif value_type == "boolean" then
        return value and "true" or "false"
    elseif value_type == "number" then
        if value ~= value then -- NaN check
            return "null"
        elseif value == math.huge then
            return "null"
        elseif value == -math.huge then
            return "null"
        else
            return tostring(value)
        end
    elseif value_type == "string" then
        return '"' .. escape_string(value) .. '"'
    elseif value_type == "table" then
        -- Check if it's an array (consecutive integer keys starting from 1)
        local is_array = true
        local max_index = 0
        local count = 0
        
        for k, v in pairs(value) do
            count = count + 1
            if type(k) ~= "number" or k ~= math.floor(k) or k < 1 then
                is_array = false
                break
            end
            max_index = math.max(max_index, k)
        end
        
        if is_array and count == max_index then
            -- Encode as array
            local parts = {}
            for i = 1, max_index do
                parts[i] = encode_value(value[i], indent_level + 1)
            end
            return "[" .. table.concat(parts, ",") .. "]"
        else
            -- Encode as object
            local parts = {}
            local i = 1
            for k, v in pairs(value) do
                if type(k) == "string" or type(k) == "number" then
                    local key_str = type(k) == "string" and k or tostring(k)
                    parts[i] = '"' .. escape_string(key_str) .. '":' .. encode_value(v, indent_level + 1)
                    i = i + 1
                end
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
    else
        -- Unsupported type, encode as string
        return '"' .. escape_string(tostring(value)) .. '"'
    end
end

-- Main encode function
function json.encode(value)
    return encode_value(value)
end

-- ============================================================================
-- JSON DECODING
-- ============================================================================

-- Skip whitespace characters
local function skip_whitespace(str, pos)
    while pos <= #str do
        local char = str:sub(pos, pos)
        if char ~= ' ' and char ~= '\t' and char ~= '\n' and char ~= '\r' then
            break
        end
        pos = pos + 1
    end
    return pos
end

-- Decode a JSON string
local function decode_string(str, pos)
    local start_pos = pos
    pos = pos + 1 -- Skip opening quote
    local result = {}
    local result_pos = 1
    
    while pos <= #str do
        local char = str:sub(pos, pos)
        
        if char == '"' then
            -- End of string
            return table.concat(result), pos + 1
        elseif char == '\\' then
            -- Escape sequence
            pos = pos + 1
            if pos > #str then
                error("Unterminated string escape sequence")
            end
            
            local escape_char = str:sub(pos, pos)
            if escape_char == '"' then
                result[result_pos] = '"'
            elseif escape_char == '\\' then
                result[result_pos] = '\\'
            elseif escape_char == '/' then
                result[result_pos] = '/'
            elseif escape_char == 'b' then
                result[result_pos] = '\b'
            elseif escape_char == 'f' then
                result[result_pos] = '\f'
            elseif escape_char == 'n' then
                result[result_pos] = '\n'
            elseif escape_char == 'r' then
                result[result_pos] = '\r'
            elseif escape_char == 't' then
                result[result_pos] = '\t'
            elseif escape_char == 'u' then
                -- Unicode escape (simplified - just copy the sequence)
                result[result_pos] = '\\u'
                pos = pos + 1
                for i = 1, 4 do
                    if pos > #str then break end
                    result[result_pos] = result[result_pos] .. str:sub(pos, pos)
                    pos = pos + 1
                end
                pos = pos - 1 -- Adjust for loop increment
            else
                result[result_pos] = escape_char
            end
            result_pos = result_pos + 1
        else
            result[result_pos] = char
            result_pos = result_pos + 1
        end
        
        pos = pos + 1
    end
    
    error("Unterminated string")
end

-- Decode a JSON number
local function decode_number(str, pos)
    local start_pos = pos
    local has_decimal = false
    local has_exponent = false
    
    -- Handle negative sign
    if str:sub(pos, pos) == '-' then
        pos = pos + 1
    end
    
    -- Parse integer part
    if pos > #str or not str:sub(pos, pos):match('%d') then
        error("Invalid number format")
    end
    
    while pos <= #str and str:sub(pos, pos):match('%d') do
        pos = pos + 1
    end
    
    -- Parse decimal part
    if pos <= #str and str:sub(pos, pos) == '.' then
        has_decimal = true
        pos = pos + 1
        
        if pos > #str or not str:sub(pos, pos):match('%d') then
            error("Invalid number format")
        end
        
        while pos <= #str and str:sub(pos, pos):match('%d') do
            pos = pos + 1
        end
    end
    
    -- Parse exponent part
    if pos <= #str and (str:sub(pos, pos) == 'e' or str:sub(pos, pos) == 'E') then
        has_exponent = true
        pos = pos + 1
        
        if pos <= #str and (str:sub(pos, pos) == '+' or str:sub(pos, pos) == '-') then
            pos = pos + 1
        end
        
        if pos > #str or not str:sub(pos, pos):match('%d') then
            error("Invalid number format")
        end
        
        while pos <= #str and str:sub(pos, pos):match('%d') do
            pos = pos + 1
        end
    end
    
    local number_str = str:sub(start_pos, pos - 1)
    local number = tonumber(number_str)
    
    if not number then
        error("Invalid number: " .. number_str)
    end
    
    return number, pos
end

-- Forward declaration
local decode_value

-- Decode a JSON array
local function decode_array(str, pos)
    local result = {}
    local result_pos = 1
    pos = pos + 1 -- Skip opening bracket
    pos = skip_whitespace(str, pos)
    
    -- Handle empty array
    if pos <= #str and str:sub(pos, pos) == ']' then
        return result, pos + 1
    end
    
    while pos <= #str do
        local value, new_pos = decode_value(str, pos)
        result[result_pos] = value
        result_pos = result_pos + 1
        pos = skip_whitespace(str, new_pos)
        
        if pos > #str then
            error("Unterminated array")
        end
        
        local char = str:sub(pos, pos)
        if char == ']' then
            return result, pos + 1
        elseif char == ',' then
            pos = skip_whitespace(str, pos + 1)
        else
            error("Expected ',' or ']' in array")
        end
    end
    
    error("Unterminated array")
end

-- Decode a JSON object
local function decode_object(str, pos)
    local result = {}
    pos = pos + 1 -- Skip opening brace
    pos = skip_whitespace(str, pos)
    
    -- Handle empty object
    if pos <= #str and str:sub(pos, pos) == '}' then
        return result, pos + 1
    end
    
    while pos <= #str do
        -- Parse key
        pos = skip_whitespace(str, pos)
        if pos > #str or str:sub(pos, pos) ~= '"' then
            error("Expected string key in object")
        end
        
        local key, new_pos = decode_string(str, pos)
        pos = skip_whitespace(str, new_pos)
        
        if pos > #str or str:sub(pos, pos) ~= ':' then
            error("Expected ':' after key in object")
        end
        
        pos = skip_whitespace(str, pos + 1)
        
        -- Parse value
        local value, value_pos = decode_value(str, pos)
        result[key] = value
        pos = skip_whitespace(str, value_pos)
        
        if pos > #str then
            error("Unterminated object")
        end
        
        local char = str:sub(pos, pos)
        if char == '}' then
            return result, pos + 1
        elseif char == ',' then
            pos = skip_whitespace(str, pos + 1)
        else
            error("Expected ',' or '}' in object")
        end
    end
    
    error("Unterminated object")
end

-- Decode any JSON value
decode_value = function(str, pos)
    pos = skip_whitespace(str, pos)
    
    if pos > #str then
        error("Unexpected end of JSON input")
    end
    
    local char = str:sub(pos, pos)
    
    if char == '"' then
        return decode_string(str, pos)
    elseif char == '{' then
        return decode_object(str, pos)
    elseif char == '[' then
        return decode_array(str, pos)
    elseif char == 't' then
        if str:sub(pos, pos + 3) == "true" then
            return true, pos + 4
        else
            error("Invalid literal")
        end
    elseif char == 'f' then
        if str:sub(pos, pos + 4) == "false" then
            return false, pos + 5
        else
            error("Invalid literal")
        end
    elseif char == 'n' then
        if str:sub(pos, pos + 3) == "null" then
            return nil, pos + 4
        else
            error("Invalid literal")
        end
    elseif char:match('[%-0-9]') then
        return decode_number(str, pos)
    else
        error("Unexpected character: " .. char)
    end
end

-- Main decode function
function json.decode(str)
    if type(str) ~= "string" then
        error("JSON decode expects a string")
    end
    
    local value, pos = decode_value(str, 1)
    pos = skip_whitespace(str, pos)
    
    if pos <= #str then
        error("Extra characters after JSON")
    end
    
    return value
end

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Pretty print JSON (for debugging)
function json.encode_pretty(value, indent)
    indent = indent or "  "
    
    local function encode_pretty_value(val, level)
        level = level or 0
        local current_indent = string.rep(indent, level)
        local next_indent = string.rep(indent, level + 1)
        local val_type = type(val)
        
        if val_type == "table" then
            -- Check if array
            local is_array = true
            local max_index = 0
            local count = 0
            
            for k, v in pairs(val) do
                count = count + 1
                if type(k) ~= "number" or k ~= math.floor(k) or k < 1 then
                    is_array = false
                    break
                end
                max_index = math.max(max_index, k)
            end
            
            if is_array and count == max_index then
                -- Array
                if count == 0 then
                    return "[]"
                end
                
                local parts = {"[\n"}
                for i = 1, max_index do
                    parts[#parts + 1] = next_indent
                    parts[#parts + 1] = encode_pretty_value(val[i], level + 1)
                    if i < max_index then
                        parts[#parts + 1] = ","
                    end
                    parts[#parts + 1] = "\n"
                end
                parts[#parts + 1] = current_indent .. "]"
                return table.concat(parts)
            else
                -- Object
                local keys = {}
                for k in pairs(val) do
                    if type(k) == "string" or type(k) == "number" then
                        keys[#keys + 1] = k
                    end
                end
                
                if #keys == 0 then
                    return "{}"
                end
                
                table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
                
                local parts = {"{\n"}
                for i, k in ipairs(keys) do
                    local key_str = type(k) == "string" and k or tostring(k)
                    parts[#parts + 1] = next_indent
                    parts[#parts + 1] = '"' .. escape_string(key_str) .. '": '
                    parts[#parts + 1] = encode_pretty_value(val[k], level + 1)
                    if i < #keys then
                        parts[#parts + 1] = ","
                    end
                    parts[#parts + 1] = "\n"
                end
                parts[#parts + 1] = current_indent .. "}"
                return table.concat(parts)
            end
        else
            return encode_value(val)
        end
    end
    
    return encode_pretty_value(value)
end

return json