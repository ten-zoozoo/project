CREATE TABLE sicu.patients (
    subject_id        BIGINT,
    hadm_id           BIGINT,
    stay_id           BIGINT,
    intime            TIMESTAMP,
    outtime           TIMESTAMP,
    intime24          TIMESTAMP,
    gender            VARCHAR,
    환자실제나이      INTEGER,
    dod_within_24h    INTEGER,
    "최초입실"          INTEGER
);

CREATE TABLE icu.inputevents (
    subject_id   BIGINT,
    hadm_id      BIGINT,
    stay_id      BIGINT,
    starttime    TIMESTAMP,
    endtime      TIMESTAMP,
    itemid       BIGINT,
    label        TEXT,
    amount       DOUBLE PRECISION,
    amountuom    TEXT
);

CREATE TABLE sicu.inputevents (
    subject_id   BIGINT,
    hadm_id      BIGINT,
    stay_id      BIGINT,
    starttime    TIMESTAMP,
    endtime      TIMESTAMP,
    itemid       BIGINT,
    label        TEXT,
    amount       DOUBLE PRECISION,
    amountuom    TEXT,
    tablename    TEXT
);

CREATE TABLE sicu.outputevents (
    subject_id   BIGINT,
    hadm_id      BIGINT,
    stay_id      BIGINT,
    starttime    TIMESTAMP,
    endtime      TIMESTAMP,
    itemid       BIGINT,
    label        TEXT,
    amount       DOUBLE PRECISION,
    amountuom    TEXT,
    tablename    TEXT
);

CREATE TABLE sicu.labevents (
    subject_id   BIGINT,
    hadm_id      BIGINT,
    stay_id      BIGINT,
    starttime    TIMESTAMP,
    endtime      TIMESTAMP,
    itemid       BIGINT,
    label        TEXT,
    amount       DOUBLE PRECISION,
    amountuom    TEXT,
    tablename    TEXT
);

CREATE TABLE sicu.chartevents (
    subject_id   BIGINT,
    hadm_id      BIGINT,
    stay_id      BIGINT,
    starttime    TIMESTAMP,
    endtime      TIMESTAMP,
    itemid       BIGINT,
    label        TEXT,
    amount       DOUBLE PRECISION,
    amountuom    TEXT,
    tablename    TEXT
);

CREATE TABLE sicu.diagnoses_icd (
    subject_id   BIGINT,
    hadm_id      BIGINT,
    seq_num      INTEGER,
    icd_code     TEXT,
    icd_version  INTEGER
);

CREATE TABLE sicu.procedures_icd (
    subject_id   BIGINT NOT NULL,
    hadm_id      BIGINT NOT NULL,
    seq_num      INTEGER,
    chartdate    DATE,
    icd_code     TEXT NOT NULL,
    icd_version  INTEGER NOT NULL
);

-- 사망률 예측 데이터 저장
CREATE TABLE mortality_prediction (
    id SERIAL PRIMARY KEY,
    stay_id BIGINT NOT NULL,
    admission_time TIMESTAMP NOT NULL,
    check_time TIMESTAMP NOT NULL,
    hours INTEGER NOT NULL CHECK (hours IN (6, 12, 18, 24)),
    pred_die_percent NUMERIC(5,1) NOT NULL,
    diff NUMERIC(5,1)
);
