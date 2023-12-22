# Changelog

All notable changes to `rollup-plugin-scss` will be documented in this file.

## [Unreleased]


## [3.0.0] - 2021-06-29

### Added

- Add insert option @syJSdev
- Add `sourceMap` option to enable generation of source map @astappiev
- Add automated testing using Github Actions

### Updated

- A `processor` can receive map as second parameter and return `{ css: string, map?: string }`
- Remove `node-sass` from optionalDependencies @astappiev <br/>
  **You have to specify `node-sass` or `sass` in your project dependencies alongside `rollup-plugin-scss`**

## [2.6.1] - 2020-10-01

### Updated

- Move node-sass to optionalDependencies @weizhenye

## [2.6.0] - 2020-08-14

### Fixed

- Resolve processor as a promise during transform step @geotrev

### Added

- Add support for postcss processor

## [2.5.0] - 2020-05-07

### Updated

- Fix includePaths before processing @mach25

## [2.4.0] - 2020-04-13

### Added

- Add `sass` option @riri

## [2.2.0] - 2020-04-11

### Added

- Add `watch` option @JimSchofield

## [2.1.0] - 2019-12-22

### Added

- Add `prefix` option @jackprosser

## [2.0.0] - 2019-12-22

### Changed

- Add `node_modules/` in includePaths by default
- Fix cases where output `css` is null or undefined
- Update dependencies

## [1.0.0] - 2019-02-04

### Update

- Update `ongenerate` to `generateBundle`

[unreleased]: https://github.com/thgh/rollup-plugin-scss/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/thgh/rollup-plugin-scss/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/thgh/rollup-plugin-scss/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/thgh/rollup-plugin-scss/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/thgh/rollup-plugin-scss/compare/v0.0.1...v1.0.0
[0.0.1]: https://github.com/thgh/rollup-plugin-scss/releases
