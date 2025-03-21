import antimony
from roadrunner import RoadRunner
from typing import Any, Dict, List, Optional, Tuple
import json


class SBMLModel:

    def __init__(self,
                 sbml: str,
                 results_names: List[str],
                 param_dists: Optional[Dict[str, Tuple[str, Any]]] = None,
                 mods: Optional[Dict[str, float]] = None) -> None:
        super().__init__()

        self.sbml = sbml
        self.results_names = results_names
        self.param_dists = param_dists
        self.mods = mods

    def to_json(self) -> dict:
        json_data = dict(sbml=self.sbml, 
                         results_names=self.results_names)
        if self.param_dists is not None:
            json_data['param_dists'] = self.param_dists
        if self.mods is not None:
            json_data['mods'] = self.mods
        return json_data

    @staticmethod
    def from_json(json_data: dict):
        param_dists = {k: (t[0], tuple([float(tt) for tt in t[1]])) for k, t in
                       json_data['param_dists'].items()} if 'param_dists' in json_data.keys() else None
        return SBMLModel(sbml=json_data['sbml'],
                         results_names=json_data['results_names'],
                         param_dists=param_dists,
                         mods=json_data['mods'] if 'mods' in json_data.keys() else None)

    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.to_json(), f)

    @staticmethod
    def load(file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)

        return SBMLModel.from_json(data)


class OnlineSBMLModel(SBMLModel):

    def __init__(self, uri: str, *args, **kwargs) -> None:

        super().__init__(RoadRunner(uri).getCurrentSBML(), *args, **kwargs)


class AntimonyModel(SBMLModel):

    def __init__(self, antimony_model: str, *args, **kwargs) -> None:

        antimony.clearPreviousLoads()
        antimony.loadAntimonyString(antimony_model)
        sbml = antimony.getSBMLString(antimony.getMainModuleName())

        super().__init__(sbml, *args, **kwargs)


###################
# Implementations #
###################


def model_bistable(*args, **kwargs):
    return AntimonyModel(
        """
        species y;

        -> y ; - y * (1 - y) * (1 - y / 2);
        -> z ; s * cos(s * (1 + y) * time) * (1 + y * (1 - (1 - y) * (1 - y / 2) * time));

        s = 5.0;
        y = 1.0;
        z = 0.0;
        """, 
        ['y', 'z'], 
        *args, **kwargs
    )


def model_bistable2(*args, **kwargs):
    return AntimonyModel(
        """
        species y;

        -> y ; - y * (a - y) * (1 - y / 2);
        -> z ; s * cos(s * (1 + y) * time) * (1 + y * (1 - (1 - y) * (1 - y / 2) * time));

        a = 1.0;
        s = 5.0;
        y = 1.0;
        z = 0.0;
        """, 
        ['y', 'z'], 
        *args, **kwargs
    )


def model_coinfection(*args, **kwargs):
    return AntimonyModel(
        """
        species T, I1, I2, V, P;

        aFunc := a * P^z;
        fFunc := n^2 * MA / (P^2 + n^2 * MA);
        phiFunc := phi * V / (KPV + V);

        -> T  ; -beta * T * V;
        -> I1 ; beta * T * V - k * I1 - mu * P * I1;
        -> I2 ; k * I1 - delta * I2 - mu * P * I2;
        -> V  ; p * I2 * (1 + aFunc) - c * V;
        -> P  ; r * P * (1 - P / KP / (1 + psi * V)) - gammaMA * fFunc * MA * P * (1 - phiFunc);

        E1: at(time>=t0): P = P0;

        T = 1E7;
        I1 = 0.0;
        I2 = 0.0;
        V = 2.0;
        P = 0.0;

        beta = 2.8E-6;
        k = 4.0;
        delta = 0.89;
        p = 25.1;
        c = 28.4;
        r = 27.0;
        KP = 2.3E8;
        gammaMA = 1.35E-4;
        n = 5.0;
        MA = 1E6;
        phi = 1E3;
        KPV = 1.8E3;
        a = 1.2E-3;
        z = 0.5;
        psi = 1.2E-8;
        mu = 5.2E-10;
        P0 = 1E3;
        t0 = 7.0;
        """,
        ['T', 'I1', 'I2', 'V', 'P'],
        *args, **kwargs
    )


def model_lines(*args, **kwargs):
    return AntimonyModel(
        """
        species y, z;

        y = 1.0;
        z = 2.0;
        """,
        ['y', 'z'],
        *args, **kwargs
    )


def model_lorentz(*args, **kwargs):
    return AntimonyModel(
        """
        species x, y, z;

        -> x ; sig * (y - x);
        -> y ; x * (rho - z) - y;
        -> z ; x * y - beta * z;

        x = 1.0;
        y = 1.0;
        z = 1.0;

        rho = 28.0;
        sig = 10.0;
        beta = 8.0 / 3;
        """,
        ['x', 'y', 'z'],
        *args, **kwargs
    )


def model_nlpendulum(*args, **kwargs):
    return AntimonyModel(
        """
        species t, v;

        -> t ; v;
        -> v ; -a * sin(t);

        a = 1.0;
        t = 0.78539816339744830961566084581988;
        v = 0.0;
        """,
        ['t', 'v'], 
        *args, **kwargs
    )


def model_oscillator(*args, **kwargs):
    return AntimonyModel(
        """
        species Y, Z;

        -> Y ; -sin(time * (1 + t0)) * (1 + t0);
        -> Z ;  cos(time * (1 + t0)) * (1 + t0);

        t0 = 0.0;
        Y = 1;
        Z = 0;
        """,
        ['Y', 'Z'],
        *args, **kwargs
    )


def model_pulse(*args, **kwargs):
    return AntimonyModel(
        """
        -> x ; 2 * n * k * (sin(k * time) ^ (2 * n - 1)) * cos(k * time) - u * x;
        -> y ; u * x - v * y;

        n = 20;
        k = 1.0;
        u = 0.9;
        v = 0.5;

        x = 0;
        y = 0;
        """,
        ['x', 'y'], 
        *args, **kwargs
    )


def model_seir(*args, **kwargs):
    return AntimonyModel(
        """
        species S, E, I, R, V;

        S -> E ; 0.00001 * S * V;
        E -> I ; 0.2 * E;
        I -> R ; 0.2 * I;
        -> V ; 10.0 * I - 0.1 * V;

        S = 1000.0;
        E = 0.0;
        I = 0.0;
        R = 0.0;
        V = 1000.0;
        """,
        ['S', 'E', 'I', 'R', 'V'],
        *args, **kwargs
    )


def model_sir(*args, **kwargs):
    return AntimonyModel(
        """
        species S, I, R, V;

        S -> I ; beta * S * V;
        I -> R ; delta * I;
        -> V  ; p * I - k * V;

        S = 1E6;
        I = 0.0;
        R = 0.0;
        V = 2.0;

        beta = 2.0E-6;
        k = 4.0;
        delta = 1E0;
        p = 25.0;
        """,
        ['S', 'I', 'R', 'V'],
        *args, **kwargs
    )


def model_tellurium_ex(*args, **kwargs):
    return AntimonyModel(
        """
        J1: S1 -> S2;  k1*S1;
        J2: S2 -> S3; k2*S2 - k3*S3
        J3: S3 -> S4; k4*S3;

        k1 = 0.1; k2 = 0.5; k3 = 0.5; k4 = 0.5;
        S1 = 100;
        """,
        ['S1', 'S2', 'S3', 'S4'],
        *args, **kwargs
    )


def biomodels_1805160001(*args, **kwargs):
    return OnlineSBMLModel(
        "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL1805160001/2/Lo2005.xml",
        [
            "TF",
            "TF_VII",
            "VII",
            "TF_VIIa",
            "VIIa",
            "Xa",
            "IIa",
            "TF_VIIa_X",
            "X",
            "TF_VIIa_Xa",
            "IX",
            "TF_VIIa_IX",
            "IXa",
            "II",
            "VIII",
            "VIIIa",
            "IXa_VIIIa",
            "IXa_VIIIa_X",
            "VIIIa_1_L",
            "VIIIa_2",
            "V",
            "Va",
            "Xa_Va",
            "Xa_Va_II",
            "mIIa",
            "TFPI",
            "Xa_TFPI",
            "TF_VIIa_Xa_TFPI",
            "ATIII",
            "Xa_ATIII",
            "mIIa_ATIII",
            "IXa_ATIII",
            "IIa_ATIII",
            "TF_VIIa_ATIII"
        ],
        *args, **kwargs
    )


def biomodels_2006170003(*args, **kwargs):
    return OnlineSBMLModel(
        "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL2006170003/4/D1_Gq_EJN_fixed.xml",
        [
            "Gqabg", 
            "mGluR", 
            "Glu", 
            "Ca1", 
            "Glu_mGluR_Gqabg", 
            "Glu_mGluR", 
            "GqaGTP", 
            "GqaGDP", 
            "Plc", 
            "IP3", 
            "Plc_Ca", 
            "Plc_Ca_GqaGTP", 
            "Pip2", 
            "Plc_Ca_Pip2", 
            "Dag", 
            "Plc_Ca_GqaGTP_Pip2", 
            "Plc_GqaGTP", 
            "DagL", 
            "Ca_DagL", 
            "Dag_Ca_DagL", 
            "TwoAG", 
            "DagK", 
            "DagK_Dag", 
            "PA", 
            "PKC", 
            "PKC_Ca", 
            "PKCt", 
            "PIkin", 
            "IP3_deg", 
            "IP3_deg_PIkin", 
            "TwoAG_degrad", 
            "Glu_mGluR_desens", 
            "Glu_out", 
            "Gqabg_mGluR", 
            "Da", 
            "Da_Out", 
            "D1R", 
            "Da_D1R", 
            "Gsabg", 
            "Da_D1R_Gsabg", 
            "GsaGTP", 
            "Gsabg_D1R", 
            "GsaGDP", 
            "Ach", 
            "M4R", 
            "Ach_M4R", 
            "Giabg", 
            "Giabg_M4R", 
            "Ach_M4R_Giabg", 
            "GiaGTP", 
            "M1R", 
            "Ach_M1R", 
            "Ach_M1R_Gqabg", 
            "M1R_Gqabg", 
            "GiaGDP", 
            "ATP", 
            "cAMP", 
            "AMP", 
            "pPDE10", 
            "PDE10", 
            "PDE10_cAMP", 
            "pPDE10_cAMP", 
            "PKAc", 
            "PP1", 
            "PKAc_PDE10", 
            "PKAc_PDE10_cAMP", 
            "PDE4", 
            "PKAc_PDE4", 
            "PDE4_cAMP", 
            "pPDE4", 
            "pPDE4_cAMP", 
            "PKAc_PDE4_cAMP", 
            "CK_Cam_Ca4", 
            "pCK_Cam_Ca4", 
            "pCK", 
            "CK_Cam_Ca4_DagL", 
            "pDagL", 
            "pCK_Cam_Ca4_DagL", 
            "pCK_Cam_Ca4_Ca_DagL", 
            "Ca_pDagL", 
            "CK_Cam_Ca4_Ca_DagL", 
            "pCK_DagL", 
            "Glu_Gbuf", 
            "Gbuf", 
            "PMCA", 
            "PMCA_Ca", 
            "ncx", 
            "ncx_Ca", 
            "Ca_Out", 
            "Leak", 
            "CaOut_Leak", 
            "PKAc_Da_D1R_Gsabg", 
            "Da_pD1R_Gsabg", 
            "Da_Dbuf", 
            "Dbuf", 
            "AC5", 
            "AC5_GiaGTP", 
            "AC5_GsaGTP", 
            "AC5_GsaGTP_GiaGTP", 
            "AC5_GsaGTP_ATP", 
            "AC5_GsaGTP_GiaGTP_ATP", 
            "PDE1", 
            "Cam_Ca4", 
            "PDE1_Cam_Ca4", 
            "PDE1_Cam_Ca4_cAMP", 
            "PP2A_BPR72", 
            "PP2A_Ca", 
            "pPDE4_PP2A_Ca", 
            "pPDE4_PP2A", 
            "pPDE10_PP2A_Ca", 
            "pPDE10_PP2A", 
            "PDE2", 
            "PDE2_cAMP", 
            "PDE2_cAMP2", 
            "Cam", 
            "CamC_Ca2", 
            "CamN_Ca2", 
            "PP2A_B56d", 
            "PKAc_PP2A_B56d", 
            "pPP2A", 
            "Epac1", 
            "Epac1_cAMP", 
            "PP2B", 
            "PP2B_Cam", 
            "PP2B_CamC_Ca2", 
            "PP2B_CamN_Ca2", 
            "PP2B_Cam_Ca4", 
            "CK", 
            "pCK_PP1", 
            "pCK_Cam_Ca4_PP1", 
            "AC1", 
            "AC1_GsaGTP", 
            "AC1_Cam_Ca4", 
            "AC1_GsaGTP_Cam_Ca4", 
            "AC1_Cam_Ca4_ATP", 
            "AC1_GsaGTP_Cam_ATP", 
            "D32", 
            "D32_PKAc", 
            "D32_p34", 
            "D32_p34_PP1", 
            "D32_p34_PP2B_Cam_Ca4", 
            "D32_p34_PP1_PP2B_Cam_Ca4", 
            "D32_p34_PP2A_B56d", 
            "D32_p34_PP2A_BPR72", 
            "D32_p34_PP1_PP2A_B56d", 
            "D32_p34_PP1_PP2A_BPR72", 
            "D32_p75", 
            "CDK5", 
            "D32_CDK5", 
            "D32_p75_PP2A_BPR72", 
            "D32_p75_PP2A_B56d", 
            "D32_p75_pPP2A", 
            "D32_p75_PP2A_Ca", 
            "D32_p75_PKAc", 
            "PKA", 
            "PKA_cAMP2", 
            "PKA_cAMP4", 
            "AIP", 
            "CK_Cam_Ca4_AIP", 
            "OA", 
            "OA_PP2A_B56d", 
            "OA_PP2A_BPR72", 
            "OA_PP2A_Ca", 
            "OA_pPP2A", 
            "Telenz_M1R", 
            "Telenz", 
            "Telenz_m1R_Gq", 
            "PD102840", 
            "PD_M4R", 
            "PD_M4R_Giabg", 
            "OA_PP1", 
            "AKAR3", 
            "PKAc_AKAR3", 
            "pAKAR3", 
            "PP1_pAKAR3", 
            "PKAr", 
            "pD1R_Gsabg", 
            "total_Cam", 
            "total_D1R", 
            "total_PP2B", 
            "total_PDE1", 
            "total_PDE4", 
            "total_PDE2", 
            "total_PP1", 
            "total_D32", 
            "total_PP2A", 
            "total_PKC", 
            "total_PKA", 
            "total_M4R", 
            "total_M1R", 
            "Total_AC5", 
            "Total_mGluR", 
            "Total_PDE10", 
            "Total_DagL", 
            "total_DagK", 
            "total_PIKinase", 
            "total_Pip2", 
            "Total_Gq", 
            "Total_Gs", 
            "Total_Gi", 
            "Total_Plc", 
            "Total_CK", 
            "Total_Epac1", 
            "Total_AC1", 
            "Ca_pDagL_Dag", 
            "Gbg", 
            "et", 
            "r", 
            "s"
        ],
        *args, **kwargs
    )


def biomodels_2004140002(*args, **kwargs):
    return OnlineSBMLModel(
        "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL2004140002/1/Liu2012.xml",
        ['X', 'V', 'Y', 'Z', 'YT', 'Mx', 'My', 'Mz'],
        *args, **kwargs
    )


def biomodels_2001130001(*args, **kwargs):
    return OnlineSBMLModel(
        "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL2001130001/1/Banerjee2008.xml",
        ['x', 'y', 'z'],
        *args, **kwargs
    )


def biomodels_6615119181(*args, **kwargs):
    return OnlineSBMLModel(
        "https://www.ebi.ac.uk/biomodels/services/download/get-files/MODEL6615119181/4/BIOMD0000000010_url.xml",
        ['MKKK', 'MKKK_P', 'MKK', 'MKK_P', 'MKK_PP', 'MAPK', 'MAPK_P', 'MAPK_PP'],
        *args, **kwargs
    )
