# Backup Policy

All production databases are backed up daily at 2:00 AM local server time. File shares used by teams are backed up nightly as well.

Backup retention is 30 days on a rolling basis: every day the oldest backup ages out and a new one is taken.

Restore tests: the infrastructure team performs a test restore every quarter and documents the result. Any failed test is treated as a high-severity incident until the backup is proven working again.

Employee responsibility: keep work files in the synced company drives, not only on your laptop's local disk. Files stored only locally are NOT backed up.

Laptop backups: the endpoint agent backs up your user folder continuously when you are online. If your laptop dies, the ServiceDesk can restore your files to a replacement within 1 business day.
