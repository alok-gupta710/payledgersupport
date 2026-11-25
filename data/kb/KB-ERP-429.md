---
doc_id: KB-ERP-429
title: "ERP Sync Error ERP-429: ERP Rate Limit Reached"
version: "1.0"
effective_date: "2026-06-01"
status: active
category: erp
source: official
---
# ERP Sync Error ERP-429: ERP Rate Limit Reached
## What it means
ERP-429 means NetSuite or QuickBooks rejected the sync because too many requests were sent to the ERP in a short time.

## How to fix it
No action is needed for a single occurrence: PayLedger retries automatically after 15 minutes. If ERP-429 keeps appearing for more than 2 hours, avoid starting manual syncs and contact support with the sync log ID.
