ALTER TABLE facts ADD COLUMN active INTEGER NOT NULL DEFAULT 1;
ALTER TABLE facts ADD COLUMN superseded_by INTEGER NULL;

CREATE INDEX IF NOT EXISTS idx_facts_category_active ON facts(category, active);
CREATE INDEX IF NOT EXISTS idx_facts_superseded_by ON facts(superseded_by);
