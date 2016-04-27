from setuptools import setup

with open('tripletembedding/requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(name='tripletembedding',
      version='0.1',
      description='',
      url='https://github.com/hrantzsch/TripletEmbedding',
      author='Hannes Rantzsch',
      author_email='hannes.rantzsch@student.hpi.de',
      license='GPLv3',
      packages=['tripletembedding'],
      zip_safe=False,

      install_requires=install_requires
      )
