print("testing ciso8601...")

import ciso8601

print(ciso8601.__version__)

# Parse an ISO 8601 formatted date string
iso_string = "2026-03-18T12:00:00Z"
parsed_date = ciso8601.parse_datetime(iso_string)

print("ciso8601 OK")
