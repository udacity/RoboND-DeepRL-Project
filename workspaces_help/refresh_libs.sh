#!/bin/bash
glewh = "/usr/include/GL/glew.h"
if [-e "$glewh"] then
  sudo apt-get install -y libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev
fi
