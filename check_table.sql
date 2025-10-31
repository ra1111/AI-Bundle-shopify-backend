-- Check if generation_progress table exists
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'generation_progress'
ORDER BY ordinal_position;

-- Alternative: List all tables
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

-- Check table structure and constraints
SHOW CREATE TABLE generation_progress;
