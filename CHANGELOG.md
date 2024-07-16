# Changelog

## [Unreleased](https://github.com/convml/convml-data/tree/HEADER)

[Full Changelog](https://github.com/convml/convml-data/compare/v0.2.1...HEAD)

*enhancements*

- Keep internal list of time for which CERES (SatCORPS) data is known to be
  missing so that we don't request these files.
  [\#21](https://github.com/convml/convml-data/pull/21)

*bugfixes*

- Fix creation of tile images from "multichannel" sources that arose from using
  `matplotlib` to create the tile image, now images are created directly using
  `PIL`.
  [\#22](https://github.com/convml/convml-data/pull/22)

- Change filename-format for GOES-16 query results yaml-file so that these can
  be written/read on windows (previously the filename-format contained `:`
  which isn't supported on windows filesystems)
  [\#43](https://github.com/convml/convml-data/pull/43)


*internal*

- Simplify the code for applying time-filters to ease adding new filters in
  future [\#22](https://github.com/convml/convml-data/pull/22)

*maintenance*

- fixes for pre-commit linting tools (incompatability changes with flake8 and
  isort with newer versions of python and importlib)
  [\#42](https://github.com/convml/convml-data/pull/42)

- Update `black` version to fix issue which arose with upstream dependency and
  switch to `conda-incubator/setup-miniconda` for CI conda (since
  `s-weigand/setup-conda` broke)
  [\#28](https://github.com/convml/convml-data/pull/28)


## [v0.2.1](https://github.com/convml/convml-data/tree/v0.2.1)

[Full Changelog](https://github.com/convml/convml-data/compare/v0.2.0...v0.2.1)

*bugfixes*

- Fix issue where satpy would return brightness temperature for some channels
  but radiance for others for GOES-16 data. Always use radiance by default from
  now on. [\#20](https://github.com/convml/convml-data/pull/20)

- Fix issue where triplet tiles generated with multiple workers in parallel
  sometimes were at repeated locations, i.e. some tiles were exactly at the
  same locations [\#19](https://github.com/convml/convml-data/pull/19)


## [v0.2.0](https://github.com/convml/convml-data/tree/v0.2.0)

[Full Changelog](https://github.com/convml/convml-data/compare/v0.1.0...v0.2.0)

*new features*

- Add option for using GOES-16 brightness temperature rather than radiance
  observations
  [\#16](https://github.com/convml/convml-data/pull/16)
  ([leifdenby](https://github.com/leifdenby))

- add convenience pipeline task for generating meta info for all tiles
  [\#17](https://github.com/convml/convml-data/pull/17)


*maintenance*

- Fixes for tile generation pipeline, ensuring that domains are cropped to
  only-just contain tiles (rather than using larger sampling domain), copy
  source-data meta attributes to tiles and make tile-image generation work for
  tiles that are constructed from brightness temperature
  [\#18](https://github.com/convml/convml-data/pull/18)

- Fixes for aux variables missed during splitting `convml-data` code from
  `convml_tt`
  [\#15](https://github.com/convml/convml-data/pull/15)
  ([leifdenby](https://github.com/leifdenby))

- Pin `satdata` package to version released on pypi. This is necessary to be
  able to put `convml-data` on pypi
  [\#14](https://github.com/convml/convml-data/pull/14)
  ([leifdenby](https://github.com/leifdenby))


## [v0.1.0](https://github.com/convml/convml-data/tree/v0.1.0)

[Full Changelog](https://github.com/convml/convml-data/compare/...v0.1.0)

- Refactor to remove all functionality from convml-data which doesn't involve
  dataset generation [\#1](https://github.com/convml/convml-data/pull/1)
  ([leifdenby](https://github.com/leifdenby))
