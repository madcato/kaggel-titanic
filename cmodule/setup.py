from distutils.core import setup, Extension

# https://docs.python.org/3/extending/building.html#building

module1 = Extension('spam',
                    sources = ['spammodule.c'])

setup (name = 'spam',
       version = '1.0',
       description = 'This is a spam package',
       ext_modules = [module1])