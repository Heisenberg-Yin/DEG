# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/yinziqi/miniconda3/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/yinziqi/miniconda3/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build

# Include any dependencies generated for this target.
include test/CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/main.dir/flags.make

test/CMakeFiles/main.dir/main.cpp.o: test/CMakeFiles/main.dir/flags.make
test/CMakeFiles/main.dir/main.cpp.o: /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/test/main.cpp
test/CMakeFiles/main.dir/main.cpp.o: test/CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/main.dir/main.cpp.o"
	cd /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/main.dir/main.cpp.o -MF CMakeFiles/main.dir/main.cpp.o.d -o CMakeFiles/main.dir/main.cpp.o -c /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/test/main.cpp

test/CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	cd /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/test/main.cpp > CMakeFiles/main.dir/main.cpp.i

test/CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	cd /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/test/main.cpp -o CMakeFiles/main.dir/main.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

test/main: test/CMakeFiles/main.dir/main.cpp.o
test/main: test/CMakeFiles/main.dir/build.make
test/main: src/libSTKQ.a
test/main: test/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable main"
	cd /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/main.dir/build: test/main
.PHONY : test/CMakeFiles/main.dir/build

test/CMakeFiles/main.dir/clean:
	cd /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build/test && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/main.dir/clean

test/CMakeFiles/main.dir/depend:
	cd /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/test /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build/test /mnt/newpart/yinziqi/clion-graph-ann-tkq/graphann-tkq/build/test/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/main.dir/depend

