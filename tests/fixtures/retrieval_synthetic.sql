-- Synthetic rows only. The test harness first applies docker/db/schema.sql.
INSERT INTO papers (corpus_id,title,abstract,year) VALUES
    (1,'BERT','Language representations',2019),
    (2,'Retrieval A','Retrieval BERT baseline experiments',2021),
    (3,'Retrieval B','Retrieval survey',2022),
    (4,'Vision model','Vision experiments',2018),
    (5,'Retrieval C','Retrieval evaluation',2023);
UPDATE papers SET embedding=(
    '[' || array_to_string(ARRAY[1.0] || array_fill(0.0, ARRAY[767]), ',') || ']'
)::vector;
INSERT INTO authors (author_id,name,normalized_name) OVERRIDING SYSTEM VALUE
VALUES (10,'Alice','alice'),(20,'Bob','bob'),(30,'Carol','carol');
INSERT INTO paper_authors (corpus_id,author_id) VALUES (1,10),(1,20),(4,20),(4,30);
INSERT INTO citations (citing_id,cited_id) VALUES (2,1),(3,1),(5,1),(2,4),(5,4);
INSERT INTO chunks (chunk_id,corpus_id,section,text,seq_no) VALUES
    ('2-0',2,'Experiments','We compare BERT as a baseline for retrieval.',0),
    ('5-0',5,'Related work','BERT is discussed in related work.',0);
UPDATE chunks SET embedding=(
    '[' || array_to_string(ARRAY[1.0] || array_fill(0.0, ARRAY[767]), ',') || ']'
)::vector;
UPDATE schema_metadata SET dataset_revision='synthetic-v1',embed_model='retrieval-test' WHERE id=1;