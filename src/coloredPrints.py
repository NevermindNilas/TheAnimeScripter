from colored import fg, attr

def green(text):
    return '%s%s%s' % (fg('green'), text, attr('reset'))

def red(text):
    return '%s%s%s' % (fg('red'), text, attr('reset'))

def yellow(text):
    return '%s%s%s' % (fg('yellow'), text, attr('reset'))

def blue(text):
    return '%s%s%s' % (fg('blue'), text, attr('reset'))