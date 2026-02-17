import os
import struct
import pandas as pd
import pyodbc
import msal

# ODBC constant for passing an access token to the SQL Server ODBC driver
SQL_COPT_SS_ACCESS_TOKEN = 1256


DEFAULT_TRAINING_SQL = r"""
WITH base AS (
    SELECT
        e.subject,
        e.[description] AS body,
        i.caseTypeCodeName AS label,
        i.incidentid,
        i.ticketnumber,
        i.createdon AS incident_createdon
    FROM email e
    INNER JOIN incident i
        ON i.incidentid = e.regardingobjectid
    CROSS APPLY (
        SELECT TOP 1
            ir.createdon AS resolution_createdon
        FROM incidentresolution ir
        WHERE ir.incidentid = i.incidentid
          AND ir.resolutiontypecodeName = 'Problem Solved'
        ORDER BY ir.createdon DESC
    ) ir_latest
    WHERE i.caseTypeCodeName IN (
        'Other',
        'Renewal',
        'customer Information Request',
        'Non-Renewal',
        'Claim',
        'Cancel Policy',
        'Mortgage-Related',
        'Update Policy',
        'Billing/Payment-Related',
        'Underwriting Memo',
        'Proof of Insurance/Documents',
        'Certificate',
        'New Policy',
        'Auto Updates',
        'Update Account',
        'Replacement Cost Estimator (RCE)',
        'Annual Review'
    )
    AND i.createdon < DATEADD(day, -30, GETDATE())
    AND i.createdon < GETDATE()
    AND e.[description] IS NOT NULL
    AND e.subject IS NOT NULL
    AND LTRIM(RTRIM(e.[description])) <> ''
    AND LTRIM(RTRIM(e.subject)) <> ''
),
ranked AS (
    SELECT
        subject,
        body,
        label,
        incidentid,
        ticketnumber,
        incident_createdon,
        ROW_NUMBER() OVER (
            PARTITION BY label
            ORDER BY incident_createdon DESC
        ) AS rn
    FROM base
)
SELECT
    subject,
    body,
    label,
    incidentid,
    ticketnumber
FROM ranked
WHERE rn <= 200
ORDER BY label, incident_createdon DESC;
"""


def _acquire_access_token_for_sql() -> str:
    """
    Uses client credentials to get an Entra token for SQL-style endpoints.
    """
    tenant_id = os.environ["AZURE_TENANT_ID"]
    client_id = os.environ["AZURE_CLIENT_ID"]
    client_secret = os.environ["AZURE_CLIENT_SECRET"]

    authority = f"https://login.microsoftonline.com/{tenant_id}"
    app = msal.ConfidentialClientApplication(
        client_id=client_id,
        authority=authority,
        client_credential=client_secret,
    )

    # Scope for SQL (ODBC token usage)
    result = app.acquire_token_for_client(scopes=["https://database.windows.net/.default"])
    token = result.get("access_token")
    if not token:
        raise RuntimeError(f"Token acquisition failed: {result.get('error')} - {result.get('error_description')}")
    return token


def _token_to_odbc_bytes(access_token: str) -> bytes:
    """
    ODBC expects a 4-byte length prefix + UTF-16LE token bytes.
    """
    token_utf16 = access_token.encode("utf-16-le")
    return struct.pack("=i", len(token_utf16)) + token_utf16


def fetch_training_df() -> pd.DataFrame:
    """
    Connects to Dataverse TDS endpoint, executes the training SQL, and returns:
    columns: subject, body, label, incidentid, ticketnumber
    """
    # e.g. "brightway.crm.dynamics.com, 5558"
    server = os.getenv("DATAVERSE_TDS_SERVER")
    if not server:
        raise RuntimeError("DATAVERSE_TDS_SERVER is not set. Provide it via docker -e DATAVERSE_TDS_SERVER=...")
    
    database = os.environ.get("DATAVERSE_TDS_DB", "dataverse")
    sql = os.environ.get("TRAINING_SQL_QUERY", DEFAULT_TRAINING_SQL)

    access_token = _acquire_access_token_for_sql()
    token_bytes = _token_to_odbc_bytes(access_token)

    conn_str = (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server={server};"
        f"Database={database};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    with pyodbc.connect(conn_str, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_bytes}) as conn:
        df = pd.read_sql(sql, conn)

    #Normalize columns first
    df.columns = [c.lower() for c in df.columns]

    required = {"subject", "body", "label"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"SQL did not return required columns {required}. Got: {list(df.columns)}")

    keep = ["subject", "body", "label", "incidentid", "ticketnumber"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise RuntimeError(f"SQL did not return required columns: {missing}. Got: {list(df.columns)}")

    return df[keep]


