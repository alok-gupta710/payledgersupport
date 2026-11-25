---
doc_id: KB-PAYFAIL-007
title: "Failed Payments and Bank Return Codes"
version: "1.3"
effective_date: "2026-08-01"
status: active
category: payments
source: official
---
# Failed Payments and Bank Return Codes
## Reading a failed payment
A failed payment shows a bank return code on the payment detail page.

## R01 insufficient funds
R01 means there were insufficient funds in the funding account. Add funds, then resubmit the payment.

## R02 account closed
R02 means the vendor's bank account is closed. Ask the vendor to add new bank details through the vendor portal.

## R03 and R04 account not found or invalid
R03 means the vendor's bank account could not be located. R04 means the account number structure is invalid. For both, confirm the vendor's bank details in Vendors > Bank Details, then resubmit the payment.

## R10 and R29 unauthorized
R10 and R29 mean the receiver reported the debit as unauthorized. Do not resubmit. These returns must be escalated to the Risk and Disputes team.
