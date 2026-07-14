# Sanitized systemd units for a multi-model llama.cpp fleet on Strix Halo.
#
# These are the live configs from the reference box with deployment-specific
# details removed: generic unit names, placeholder paths, no API keys
# (units bind 127.0.0.1; put auth in front with a proxy).
#
# Placeholders to replace:
#   /opt/llama/bin/llama-server-rocm  - your ROCm build (docs/BUILD.md)
#   /opt/llama/bin/llama-server       - CPU-only build (no HIP needed)
#   /opt/models/...                   - your model paths
#   /etc/llama-fleet.env              - optional shared env file (created by you)
#   /var/lib/llama/hotcache/<seat>/   - slot save path (create, chown llama)
#   User/Group                        - a dedicated service user is good hygiene;
#                                       needs `render` + `video` groups for /dev/kfd
#
# The MemoryMax=/MemoryLow= budgets are the actual fleet values. They are
# load-bearing on a unified-memory APU - see docs/PITFALLS.md before changing
# them.
