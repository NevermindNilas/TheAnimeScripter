# prettify-xml
[![Build Status](https://img.shields.io/travis/jonathanewerner/prettify-xml.svg?style=flat-square)](https://travis-ci.org/jonathanewerner/prettify-xml)
[![Code Coverage](https://img.shields.io/codecov/c/github/jonathanewerner/prettify-xml.svg?style=flat-square)](https://codecov.io/github/jonathanewerner/prettify-xml)
[![version](https://img.shields.io/npm/v/prettify-xml.svg?style=flat-square)](http://npm.im/prettify-xml)
[![downloads](https://img.shields.io/npm/dm/prettify-xml.svg?style=flat-square)](http://npm-stat.com/charts.html?package=prettify-xml&from=2015-08-01)
[![MIT License](https://img.shields.io/npm/l/prettify-xml.svg?style=flat-square)](http://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg?style=flat-square)](http://commitizen.github.io/cz-cli/)

 > Pretty print xml.

This is a small package that synchronously pretty prints XML/HTML.

## Usage
```js
const prettifyXml = require('prettify-xml')

const input = '<div><p>foo</p><p>bar</p></div>'

const expectedOutput = [
  '<div>',
  '  <p>foo</p>',
  '  <p>bar</p>',
  '</div>',
].join('\n')

const options = {indent: 2, newline: '\n'} // 2 spaces is default, newline defaults to require('os').EOL
const output = prettifyXml(input, options) // options is optional

assert(output === expectedOutput)
```


