# VPN Setup Guide

You need the VPN to reach internal tools (dashboards, code repos, HR portal) from outside the office network.

Step 1: install the WireGuard client from the IT software portal on your laptop or phone.

Step 2: log in to the IT portal at it.northwind.example, go to "My VPN profile", and download your personal configuration file. Never share this file.

Step 3: import the configuration file into the WireGuard client and toggle the connection on. You should see a green "Active" status.

Step 4: verify by opening the internal dashboard. If it loads, you are connected.

Troubleshooting: if the tunnel connects but nothing loads, check that your MFA is approved on your phone, then restart the WireGuard client. If it still fails, open a ticket with the ServiceDesk and mention error code if one appears.

Disconnect the VPN when you are done working remotely.
