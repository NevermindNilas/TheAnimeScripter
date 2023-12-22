# babel-plugin-transform-scss

Babel plugin for converting scss or sass imports to css and embedding them in the head of the document

This plugin completely replaces the standard webpack sass-loader. This part of code can be removed:

```javascript
module: {
  rules: [
    //rules
    {
      test: /\.s[ac]ss$/i,
      use: ['style-loader', 'css-loader', 'sass-loader']
    }
  ]
}
```

### Installation

`npm install --save-dev babel-plugin-transform-scss` or `yarn add -D babel-plugin-transform-scss`

Use it in your config file:

```json
{
  "presets": [],
  "plugins": [
    "babel-plugin-transform-scss"
  ]
}
```

### How it works

This plugin looks for import of `sass` or `scss` files, when imports are found - imported files replaces with function which renders pure css to the head of the document.

```javascript
import React from 'react'
import 'style.scss'

const Button = () => {
  return (
    <div className="button">
      Custom button
    </div>
  )
}

export default Button
```

`import "style.scss"` will be transpiled to css and will be rendered to the `head` of the `document`:

```html
<style data-scss-component="Button_style">
    .button {
      display: flex;
      justify-content: space-around;
    }
</style>
```

if import contains more than one styles:


##### Input
```javascript
import React from 'react'
import 'style.scss'
import 'style2.scss'

const Button = () => {
  return (
    <div className="button">
      <div className="nested-from-style2">Nested</div>
    </div>
  )
}

export default Button
```
##### Output
```html
<style data-scss-component="Button_style">
    .button {
      display: flex;
      justify-content: space-around;
    }
</style>
<style data-scss-component="Button_style2">
    .nested-from-style2 {
      display: grid;
    }
</style>
```
