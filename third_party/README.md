# Vendored header-only libraries

All chosen from [p-ranav/awesome-hpp](https://github.com/p-ranav/awesome-hpp), MIT-licensed only.
Vendored (rather than fetched via CMake `FetchContent`) to keep the Podman build hermetic
against upstream network/asset changes. To re-vendor after bumping a pinned version, re-run
the matching `curl` command below and update the version/date here.

| Library | Version | File | License | Source |
|---|---|---|---|---|
| [argparse](https://github.com/p-ranav/argparse) | v3.2 | `argparse/argparse.hpp` | MIT | `raw.githubusercontent.com/p-ranav/argparse/v3.2/include/argparse/argparse.hpp` |
| [nanobench](https://github.com/martinus/nanobench) | v4.3.11 | `nanobench/nanobench.h` | MIT | `raw.githubusercontent.com/martinus/nanobench/v4.3.11/src/include/nanobench.h` |
| [nlohmann/json](https://github.com/nlohmann/json) | v3.12.0 | `json/json.hpp` | MIT | `raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp` |
| [doctest](https://github.com/doctest/doctest) | v2.4.8 | `doctest/doctest.h` | MIT | `raw.githubusercontent.com/doctest/doctest/v2.4.8/doctest/doctest.h` |
| [magic_enum](https://github.com/Neargye/magic_enum) | v0.9.8 | `magic_enum/magic_enum.hpp` | MIT | `raw.githubusercontent.com/Neargye/magic_enum/v0.9.8/include/magic_enum/magic_enum.hpp` |

Vendored 2026-07-06.
