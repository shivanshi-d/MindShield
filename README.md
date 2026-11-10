----------------------------------------------------------
# Project Name

A fully on-device Chrome extension that flags manipulative Terms of Service clauses and dark patterns as you browse, for students and families. 
The capstone version adds a parent companion app that shows only the single most severe dark-pattern category a child ran into.

## The client

Organization: FairPlay board member (open)
Signs the agreement:
Primary user (who operates it day to day): students/parents

How they do this today, without software: There is no previous app


## The plan

Stack: JavaScript Chrome extension; two TF-IDF + logistic regression
classifiers reimplemented in JS (ToS clauses trained on CLAUDETTE, plus a
dark-pattern classifier); no backend. Companion app stack: [TBD]
Where the data lives: On the child's device. The companion app receives only
the category of the single most severe dark pattern, never URLs, browsing
history or page content, and the child knows and consents.
Where it will be deployed: Chrome Web Store (already submitted); companion app: [TBD]
My one new axis: new domain (consent-based parent–child sharing)


## Status

Week 1. Nothing built yet.

## Team
- Shivanshi
----------------------------------------------------------
