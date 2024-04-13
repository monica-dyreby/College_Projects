-- inject.lua - ion trap demo program for injection of ions into trap.
--
-- D.Manura-2006-08 - based on PRG code from SIMION 7.0 - David A. Dahl 1995
-- (c) 2006 Scientific Instrument Services, Inc. (Licensed under SIMION 8.0)
--=======================================================================
simion.workbench_program()

print "NOTE: Before use, set TQual = 0, enable "
print "Grouped flying, set Repulsion to None, and"
print "enable Dots flying at maximum speed (use slider)."

local TRAP = simion.import("traputil.lua")
TRAP.randomize_x = false -- prevent randomization in x position.

local SIMPLE = simion.import("collision_simple.lua")

-- override traputil.lua.

adjustable cone_angle_off_vel_axis = 45.0
adjustable random_offset_mm        = 1.0

-- override collision_simple.lua
adjustable _mean_free_path_mm      = 1.0


TRAP.install_segments {
    TRAP.segment,
    SIMPLE.segment,
    inject_segment
}
