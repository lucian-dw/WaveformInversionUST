/**
 * @file      TimeMeasure.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file for class providing time measurement.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      15 August    2012, 09:35 (created) \n
 *            11 February  2020, 16:24 (revised)
 *
 * @copyright Copyright (C) 2012 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * k-Wave is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#ifndef TIME_MEASURE_H
#define TIME_MEASURE_H

#include <chrono>

/// Monotonic elapsed wall time, including resumed simulation legs.
class TimeMeasure
{
  public:
    void start() { mStartTime = Clock::now(); }
    void stop() { mStopTime = Clock::now(); }
    double getElapsedTime() const
    {
      return std::chrono::duration<double>(mStopTime - mStartTime).count();
    }
    double getElapsedTimeOverAllLegs() const
    {
      return mElapsedTimeOverPreviousLegs + getElapsedTime();
    }
    double getElapsedTimeOverPreviousLegs() const
    {
      return mElapsedTimeOverPreviousLegs;
    }
    void SetElapsedTimeOverPreviousLegs(double seconds)
    {
      mElapsedTimeOverPreviousLegs = seconds;
    }

  private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point mStartTime{};
    Clock::time_point mStopTime{};
    double mElapsedTimeOverPreviousLegs = 0;
};

#endif
