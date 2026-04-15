"""Helper functions for Taylor-Couette simulation in OpenFOAM

At each time step 
1. The sim will be paused
2. Omega will be changed
3. The sim will continue with updated omega
"""

import numpy as np
from pathlib import Path
import subprocess

class Helpers():
    def __init__(self, case_path):
        self.case_path = case_path

    def _get_latest_time(self):
        """Gets the latest simulation time folder"""
        current_t = -1.0
        current_name = "0"
        for p in Path(self.case_path).iterdir():
            try:
                t = float(p.name)
                if t > current_t:
                    current_t = t
                    current_name = p.name
            except ValueError:
                pass
        return current_name
    
    def _set_omega(self, chosen_omega):
        """Changes the set angular velocity in the OpenFOAM case"""
        latest_time = self._get_latest_time()
        # Uses foamDictionary which is a command line tool for OpenFOAM
        subprocess.run(
            ["foamDictionary",
             "-entry", "boundaryField.inner_wall.omega",
             "-set", str(chosen_omega),
             f"{latest_time}/U"],
             cwd=self.case_path, check=True
        )
        return True

    def _parse_metrics(self, line):
        """Parses RL_metrics log line from system/conrolDict"""
        parts = dict(p.split("=") for p in line.split()[1:])
        return {k: float(v) for k, v in parts.items()}
    
    def _update_end_time(self, time_step):
        """Updates the endTime in controlDict to continue for (time_step) more seconds"""
        latest_time = float(self._get_latest_time())
        new_end = latest_time + time_step
        subprocess.run(
            ["foamDictionary",
             "-entry", "endTime",
             "-set", str(new_end),
             "system/controlDict"],
             cwd=self.case_path, check=True
        )
        return True
    
    def do_simulation(self, chosen_omega, time_step):
        """Runs pimpleFoam for (time_step) seconds with chosen angular velocity.
        
        Returns a list of the metrics for each time interval of the step
        {time, Mz_kin, concentrations}
        """
        self._set_omega(chosen_omega)
        self._update_end_time(time_step)
        result = subprocess.run(
            ["pimpleFoam"],
             cwd=self.case_path, check=True, capture_output=True, text=True
        )
        metric_lines= [l for l in result.stdout.splitlines() if l.startswith("METRICS")]
        if not metric_lines:
            raise RuntimeError(f"No METRICS in pimpleFoam output:\n{result.stderr[-500:]}")
        step_metrics = [self._parse_metrics(l) for l in metric_lines]
        return step_metrics
    
        

