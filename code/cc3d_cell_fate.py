"""
This simulates multiple instances of the stochastic boolean network "Cell Fate" provided at https://maboss.curie.fr.
It requires the "cc3d" conda package available on the "compucell3d" conda channel.
"""

import argparse
from cc3d.core import PyCoreSpecs as pcs
from cc3d.core.PySteppables import SteppableBasePy
from cc3d.core.simservice.CC3DSimService import CC3DSimService
from cc3dext.MaBoSSCC3DPy import CC3DMaBoSSEngineContainer
import json
import os
from random import randint

maboss_bnd = '''
node FASL
{
  rate_up = 0.0;
  rate_down = 0.0;
}
node TNF
{
  rate_up = 0.0;
  rate_down = 0.0;
}
node TNFR
{
  logic = TNF;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0;
}
node DISC_TNF
{
  logic = FADD & TNFR;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0;
}
node DISC_FAS
{
  logic = FASL & FADD;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0;
}
node FADD
{
  rate_up = 0.0;
  rate_down = 0.0 + 1000*$FADD_del;
}
node RIP1
{
  logic = (DISC_FAS | TNFR) & (!CASP8);
  rate_up = (@logic & (!$RIP1_del)) ? 1.0 : 0.0;
  rate_down = (@logic ? 0.0 : 1.0) + 1000*$RIP1_del;
}
node RIP1ub
{
  logic = cIAP & RIP1;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0;
}
node RIP1K
{
  logic = RIP1;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0;
}
node IKK
{
  logic = RIP1ub;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0;
}
node NFkB
{
  logic = IKK & (!CASP3);
  rate_up = ((@logic & (!$NFkB_del)) ? 1.0 : 0.0) + 1000*$NFkB_oe;
  rate_down = ((@logic | $NFkB_oe) ? 0.0 : 1.0) + 1000*$NFkB_del;
}
node CASP8
{
  logic = (DISC_TNF | (DISC_FAS | CASP3) ) & (!cFLIP);
  rate_up = (@logic & (!$CASP8_del)) ? 1.0 : 0.0;
  rate_down = (@logic ? 0.0 : 1.0) + 1000*$CASP8_del;
}
node BAX
{
  logic = CASP8 & (!BCL2);
  rate_up = (@logic & (!$BAX_del)) ? 1.0 : 0.0;
  rate_down = (@logic ? 0.0 : 1.0) + 1000*$BAX_del;
}
node BCL2
{
  logic = NFkB;
  rate_up = (@logic ? $TransRate : 0.0) + 1000*$BCL2_oe;
  rate_down = (@logic | $BCL2_oe) ? 0.0 : 1.0; 
}
node ROS
{
  logic = (!NFkB) & (MPT | RIP1K );
  rate_up = @logic ? $TransRate : 0.0;
  rate_down = @logic ? 0.0 : 1.0; 
}
node ATP
{
  logic = !MPT;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0; 
}
node MPT
{
  logic = (!BCL2) & ROS;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0; 
}
node MOMP
{
  logic = BAX | MPT;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0; 
}
node SMAC 
{
  logic = MOMP;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0; 
}
node cIAP
{
  rate_up = ((NFkB & (!SMAC)) & (!$cIAP_del)) ? $TransRate : 0.0;
  rate_down = ((SMAC) ? 1.0 : 0.0) + 1000*$cIAP_del;
}
node Cyt_c
{
  logic = MOMP;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0; 
}
node XIAP
{
  logic = (!SMAC) & NFkB;
  rate_up = (@logic & (!$XIAP_del)) ? $TransRate : 0.0;
  rate_down = (@logic ? 0.0 : 1.0) + 1000*$XIAP_del; 
}
node apoptosome
{
  logic = Cyt_c & (ATP & (!XIAP));
  rate_up = (@logic & (!$APAF_del)) ? 1.0 : 0.0;
  rate_down = (@logic ? 0.0 : 1.0) + 1000*$APAF_del ;
}
node CASP3
{
  logic = apoptosome & (!XIAP);
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0;
}
node cFLIP
{
  logic = NFkB ;
  rate_up = (@logic & (!$cFLIP_del))  ? $TransRate : 0.0;
  rate_down = (@logic ? 0.0 : 1.0) + 1000*$cFLIP_del; 
}
node NonACD
{
  logic = !ATP;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0; 
}
node Apoptosis
{
  logic = CASP3;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0; 
}
node Survival
{
  logic = NFkB;
  rate_up = @logic ? 1.0 : 0.0;
  rate_down = @logic ? 0.0 : 1.0; 
}
'''

maboss_cfg = '''
TNF.istate = TRUE;
ATP.istate = TRUE;
FADD.istate = TRUE;
cIAP.istate = TRUE;

FASL.istate = FALSE ;
TNFR.istate = FALSE ;
DISC_TNF.istate = FALSE ;
DISC_FAS.istate = FALSE ;
RIP1.istate = FALSE ;
RIP1ub.istate = FALSE ;
RIP1K.istate = FALSE ;
IKK.istate = FALSE ;
NFkB.istate = FALSE ;
CASP8.istate = FALSE ;
BAX.istate = FALSE ;
BCL2.istate = FALSE ;
ROS.istate = FALSE ;
MPT.istate = FALSE ;
MOMP.istate = FALSE ;
SMAC.istate = FALSE ;
Cyt_c.istate = FALSE ;
XIAP.istate = FALSE ;
apoptosome.istate = FALSE ;
CASP3.istate = FALSE ;
cFLIP.istate = FALSE ;
NonACD.istate = FALSE ;
Apoptosis.istate = FALSE ;
Survival.istate = FALSE ;

$FADD_del = FALSE ;
$RIP1_del = FALSE ;
$NFkB_del = FALSE ;
$NFkB_oe = FALSE ;
$CASP8_del = FALSE ;
$BAX_del = FALSE ;
$BCL2_oe = FALSE ;
$cIAP_del = FALSE ;
$XIAP_del = TRUE ;
$APAF_del = FALSE ;
$cFLIP_del = FALSE ;
$TransRate = 0.1;


TNF.is_internal = TRUE ;
ATP.is_internal = TRUE ;
FADD.is_internal = TRUE ;
cIAP.is_internal = TRUE ;

FASL.is_internal = TRUE ;
TNFR.is_internal = TRUE ;
DISC_TNF.is_internal = TRUE ;
DISC_FAS.is_internal = TRUE ;
RIP1.is_internal = TRUE ;
RIP1ub.is_internal = TRUE ;
RIP1K.is_internal = TRUE ;
IKK.is_internal = TRUE ;
NFkB.is_internal = TRUE ;
CASP8.is_internal = TRUE ;
BAX.is_internal = TRUE ;
BCL2.is_internal = TRUE ;
ROS.is_internal = TRUE ;
MPT.is_internal = TRUE ;
MOMP.is_internal = TRUE ;
SMAC.is_internal = TRUE ;
Cyt_c.is_internal = TRUE ;
XIAP.is_internal = TRUE ;
apoptosome.is_internal = TRUE ;
CASP3.is_internal = TRUE ;
cFLIP.is_internal = TRUE ;


sample_count = 80000;
max_time = 100;
time_tick = 0.1;
discrete_time = 0;
use_physrandgen = FALSE;
seed_pseudorandom = 300;
display_traj = FALSE;


thread_count = 4;

statdist_traj_count = 100;
statdist_cluster_threshold = 0.8;
'''


output_data = dict(cell_ids=[], steps=[], cell_data=[])


class ReporterSteppable(SteppableBasePy):

    output_nodes = ['NonACD', 'Apoptosis', 'Survival']

    def __init__(self, _num_sims: int):

        super().__init__(frequency=1)

        self.num_sims = _num_sims
        self.models = []
        self.engine_container = CC3DMaBoSSEngineContainer()

    def start(self):
        cell_data = {}
        for i in range(self.num_sims):
            model = self.maboss_model(bnd_str=maboss_bnd,
                                      cfg_str=maboss_cfg,
                                      time_tick=0.1,
                                      seed=randint(0, int(1E6)))
            output_data['cell_ids'].append(i)
            cell_data[i] = ReporterSteppable.cell_data(model)
            self.models.append(model)
            self.engine_container.addEngine(model, i, 'cell_fate')
        output_data['steps'].append(0)
        output_data['cell_data'].append(cell_data)

    def step(self, mcs):
        self.engine_container.step()

        cell_data = {}
        for i, model in enumerate(self.models):
            cell_data[i] = ReporterSteppable.cell_data(model)
        output_data['steps'].append(mcs+1)
        output_data['cell_data'].append(cell_data)

    @staticmethod
    def cell_data(model):
        return {n: float(model[n].state) for n in ReporterSteppable.output_nodes}


def main(num_steps: int, num_sims: int, output_dir: str):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    specs = [
        pcs.Metadata(debug_output_frequency=int(1E6)),
        pcs.PottsCore(dim_x=50, dim_y=50),
        pcs.CellTypePlugin('Cell')  # CC3D requires this, though it is unused
    ]
    steppable = ReporterSteppable(num_sims)

    sim = CC3DSimService()
    sim.register_specs(specs)
    sim.register_steppable(steppable)
    sim.run()
    sim.init()
    sim.start()
    [sim.step() for _ in range(num_steps)]
    sim.finish()

    with open(os.path.join(output_dir, 'output.json'), 'w') as f:
        json.dump(output_data, f, indent=4)


class ArgParser(argparse.ArgumentParser):

    def __init__(self):

        super().__init__()

        self.add_argument('-c', '--num-sims',
                          required=True,
                          type=int,
                          dest='num_sims',
                          help='Number of cell simulations')

        self.add_argument('-s', '--num-steps',
                          required=False,
                          type=int,
                          default=50,
                          dest='num_steps',
                          help='Number of simulation steps')

        self.add_argument('-o', '--output-dir',
                          required=False,
                          type=str,
                          default=os.path.join(os.path.dirname(__file__), 'results'),
                          dest='output_dir',
                          help='Data output directory')

        self.parsed_args = self.parse_args()

    @property
    def num_sims(self) -> int:
        return self.parsed_args.num_sims

    @property
    def num_steps(self) -> int:
        return self.parsed_args.num_steps

    @property
    def output_dir(self) -> str:
        return self.parsed_args.output_dir


if __name__ == '__main__':
    pa = ArgParser()
    main(num_steps=pa.num_steps,
         num_sims=pa.num_sims,
         output_dir=pa.output_dir)
