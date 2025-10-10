# Euro-CORDEX Example Data

This folder expects two example NetCDF files provided.  
They illustrate the standard Euro-CORDEX format used for precipitation (`pr`) and temperature (`tas`).

## Files

| Variable | File |
|-----------|------|
| Precipitation | `pr_EUR-11_ECMWF-ERAINT_evaluation_r11i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_day_1980-1981.nc` |
| Temperature   | `tas_EUR-11_ECMWF-ERAINT_evaluation_r11i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_day_1980-1981.nc` |

Each file:
- Covers the **Euro-CORDEX domain** (424 × 412 grid points)
- Uses a **curvilinear grid** (`rotated_latitude_longitude`)
- Contains **daily values for 1980–1981** (731 timesteps)
- Follows the CORDEX variable naming convention ([details](https://is-enes-data.github.io/CORDEX_variables_requirement_table.pdf))

## Notes

The data illustrate the typical structure and metadata (units, missing values, etc.) used in Euro-CORDEX NetCDF files.  
Files are too large for GitHub and should be downloaded from the location provided separately (sent via E-Mail)
