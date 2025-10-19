# Sccache CI Quick Reference

## Quick Facts

| Aspect | Details |
|--------|---------|
| **Current Version** | 0.7.7 |
| **Baseline Job** | `sccache-baseline-tests` |
| **Enabled Job** | `sccache-enabled-tests` |
| **Platforms** | Ubuntu, macOS, Windows |
| **Build Type** | Release (C++17) |
| **CMake Flag** | `-DCMAKE_CXX_COMPILER_LAUNCHER=sccache` |

## CI Job Names

### Baseline (No Sccache)
- `Sccache Baseline - ubuntu-latest - Release`
- `Sccache Baseline - macos-latest - Release`
- `Sccache Baseline - windows-latest - Release`

### Enabled (With Sccache)
- `Sccache Enabled - ubuntu-latest - Release`
- `Sccache Enabled - macos-latest - Release`
- `Sccache Enabled - windows-latest - Release`

## Key Files

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Main CI workflow with sccache jobs |
| `docs/SCCACHE_CI_INTEGRATION.md` | Detailed integration guide |
| `docs/SCCACHE_QUICK_REFERENCE.md` | This file |

## Sccache Installation Locations

| Platform | Path |
|----------|------|
| Ubuntu | `~/.local/bin/sccache` |
| macOS | `~/.local/bin/sccache` |
| Windows | `%USERPROFILE%\.local\bin\sccache.exe` |

## Cache Directories

| Platform | Directory |
|----------|-----------|
| Ubuntu | `~/.cache/sccache` |
| macOS | `~/Library/Caches/sccache` |
| Windows | `~/AppData/Local/sccache` |

## Common Tasks

### Update Sccache Version

Edit `.github/workflows/ci.yml`:
```yaml
env:
  SCCACHE_VERSION: "0.8.0"  # Change this
```

### View Build Metrics

Check CI job output for:
```
=== Sccache Baseline Build Metrics ===
Build Duration: XXX seconds

=== Sccache Enabled Build Metrics ===
Build Duration: XXX seconds
```

### Calculate Performance Improvement

```
Improvement % = ((Baseline - Enabled) / Baseline) * 100
```

Example:
- Baseline: 120 seconds
- Enabled: 85 seconds
- Improvement: ((120 - 85) / 120) * 100 = 29.2%

### Check Sccache Stats

Look for in job output:
```
sccache --show-stats
```

Shows cache hits, misses, and statistics.

## Troubleshooting Checklist

- [ ] Sccache installation succeeded (check job logs)
- [ ] CMake configuration includes `-DCMAKE_CXX_COMPILER_LAUNCHER=sccache`
- [ ] Sccache cache directory is being cached between runs
- [ ] Build completes successfully with sccache enabled
- [ ] Tests pass with sccache enabled
- [ ] Build metrics are reported

## Expected Results

### First Run
- Baseline: ~120 seconds (example)
- Enabled: ~125 seconds (cache population overhead)

### Subsequent Runs
- Baseline: ~120 seconds (no cache)
- Enabled: ~60-85 seconds (cache hits)

### Improvement Trend
- Run 1: -5% (cache population)
- Run 2: +30% (cache hits)
- Run 3+: +30-50% (consistent cache hits)

## Performance Factors

**Positive Factors**:
- Large number of compilation units
- Consistent compiler flags
- Stable source code (few changes)
- Good cache hit rate

**Negative Factors**:
- Frequent compiler flag changes
- Frequent source code changes
- Small number of compilation units
- Cache misses due to environment differences

## Related Documentation

- [Full Sccache Integration Guide](SCCACHE_CI_INTEGRATION.md)
- [CI/CD Pipeline Documentation](CI_CD_PIPELINE.md)
- [Sccache GitHub](https://github.com/mozilla/sccache)

## Support

For issues or questions:
1. Check the full integration guide
2. Review CI job logs for error messages
3. Check sccache GitHub issues
4. Consult the XSigma team

