import numpy as np
from numpy.linalg import norm, inv, det
import os
import pickle
import warnings
import sys


from collections import OrderedDict

from itertools import combinations
from math import cos, sin, acos, asin, pi, sqrt, tan, exp
from rep_utils import *
from SG_utils import *

from scipy.linalg import null_space, sqrtm

from itertools import permutations, chain
from sympy import sqrt, I, Integer,MatrixSymbol
from sympy import symbols, Mul
from sympy.matrices import Matrix
from sympy.matrices import  eye,zeros
from sympy.physics.quantum import TensorProduct
import argparse
import json

class ssgGroup:  # ssg group
    def __init__(self, ssgNum, group_type):
        assert group_type == 'double' or group_type == 'single'
        self.group_type = group_type
        self.ssgNum =ssgNum  # ssg number
        self.kvec = [] # for translation factor
        # bz
        self.b1 = []
        self.b2 = []
        self.b3 = []
        self.pure_T = []
        # all operations :
        self.superCell = []
        self.Gid = ''
        self.rotC = []  # the rotation part of all operations as a list
        self.tauC = []  # the translation vector of all operations as a list
        self.spin = []
        self.su2s = []  # the SU2 matrices
        self.time_reversal = [] # 1 and -1 for time reversal
        # anti-unitary operation (can be choosen arbitrary)
        self.anti_spin = []
        self.anti_rotC = []
        self.anti_tau = []
        #
        self.mul_table = []
        self.factor_su2 = []
        self.factor_trans = []

    def load_ssg(self, ssg_dic):
        def uni_or_anti(ssg_dic):
            URot = ssg_dic['URot'][-1]
            QRot = ssg_dic['QRotC']
            QTau = ssg_dic['QTauC']
            uni = 1 # -1 if anti-unitary
            for i, spin in enumerate(URot):
                if det(spin) < 0:
                    uni = -1
                    return [uni, spin, QRot[i], QTau[i]]
            return [uni]
        self.superCell = ssg_dic['superCell']
        self.Gid = ssg_dic['Gid']
        HRot = ssg_dic['HRotC']
        HTau = ssg_dic['HTauC']
        QRot = ssg_dic['QRotC']
        QTau = ssg_dic['QTauC']
        URot = ssg_dic['URot'][-1]
        # print_matlist(URot)
        # print_matlist(QTau)
        uni = uni_or_anti(ssg_dic)
        if uni[0] == -1:
            A_spin = uni[1]
            A_rot = uni[2]
            A_tau = uni[3]
            self.anti_spin = A_spin
            self.anti_rotC = A_rot
            self.anti_tau = A_tau
        else:
            self.anti_spin = 0
            self.anti_rotC = 0
            self.anti_tau = 0
        # generate a new element list
        space_rot_uni = []
        space_tau_uni = []
        spin_uni = []
        su2_uni = []
        space_rot_tot = []
        space_tau_tot = []
        spin_tot = []
        su2_tot = []
        time_reversal = []
        for i, hrot in enumerate(HRot):
            htau = HTau[i]
            for j, spin in enumerate(URot):
                if det(spin) > 0:
                    qrot = QRot[j]
                    qtau = QTau[j]
                    # r1 t1 * r2 t2 = r1r2| r1t2 + t1
                    # qrot|qtau * hrot|htau = qrot*hrot| qrot*htau + qtau
                    rot_new = qrot @ hrot
                    tau_new = qrot @ htau + qtau
                    su2_new = SU2(spin)
                    space_rot_uni.append(rot_new)
                    space_tau_uni.append(tau_new)
                    spin_uni.append(spin)
                    su2_uni.append(su2_new)
                    time_reversal.append(1)
                    space_rot_tot.append(rot_new)
                    space_tau_tot.append(tau_new)
                    spin_tot.append(spin)
                    su2_tot.append(su2_new)

        # here the space_rot, space_tau, spin, su2 are the unitary subgroup of an antiunitary group
        # use the antiunitary element A to generate all the elements G=H+AH
        if uni[0] == -1:
            Time_R = -1j * np.array([[0, -1j], [1j, 0]])
            A_su2 = Time_R @ np.conj(SU2(-A_spin))
            for i,rot in enumerate(space_rot_uni):
                tau = space_tau_uni[i]
                spin = spin_uni[i]
                su2 =  su2_uni[i]
                space_rot_tot.append(A_rot @ rot)
                space_tau_tot.append(A_rot @ tau + A_tau)
                spin_tot.append(A_spin @ spin)
                su2_tot.append(A_su2 @ np.conj(su2))
                time_reversal.append(-1)
        self.rotC = space_rot_tot
        self.tauC = space_tau_tot
        self.spin = spin_tot
        self.su2s = su2_tot
        self.time_reversal = time_reversal

    def load_ssg_2d(self, ssg_dic):
        self.anti_spin = np.array([[1,0,0],[0,1,0],[0,0,-1]])
        self.anti_rotC = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.anti_tau = np.array([0,0,0])
        self.superCell = ssg_dic['superCell']
        self.Gid = ssg_dic['Gid']
        HRot = ssg_dic['HRotC']
        HTau = ssg_dic['HTauC']
        QRot = ssg_dic['QRotC']
        QTau = ssg_dic['QTauC']
        URot = ssg_dic['URot'][-1]
        # uni = uni_or_anti(ssg_dic)
        # assert uni[0] == -1
        # A_spin = uni[1]
        # A_rot = uni[2]
        # A_tau = uni[3]
        # generate a new element list
        space_rot_uni = []
        space_tau_uni = []
        spin_uni = []
        su2_uni = []
        space_rot_tot = []
        space_tau_tot = []
        spin_tot = []
        su2_tot = []
        time_reversal = []
        for i, hrot in enumerate(HRot):
            htau = HTau[i]
            for j, spin in enumerate(URot):
                if det(spin) > 0:
                    qrot = QRot[j]
                    qtau = QTau[j]
                    # r1 t1 * r2 t2 = r1r2| r1t2 + t1
                    # qrot|qtau * hrot|htau = qrot*hrot| qrot*htau + qtau
                    rot_new = qrot @ hrot
                    tau_new = qrot @ htau + qtau
                    su2_new = SU2(spin)
                else:
                    spin = np.array([[1,0,0],[0,1,0],[0,0,-1]]) @ spin
                    qrot = QRot[j]
                    qtau = QTau[j]
                    # r1 t1 * r2 t2 = r1r2| r1t2 + t1
                    # qrot|qtau * hrot|htau = qrot*hrot| qrot*htau + qtau
                    rot_new = qrot @ hrot
                    tau_new = qrot @ htau + qtau
                    su2_new = SU2(spin)
                space_rot_uni.append(rot_new)
                space_tau_uni.append(tau_new)
                spin_uni.append(spin)
                su2_uni.append(su2_new)
                time_reversal.append(1)
                space_rot_tot.append(rot_new)
                space_tau_tot.append(tau_new)
                spin_tot.append(spin)
                su2_tot.append(su2_new)

        # here the space_rot, space_tau, spin, su2 are the unitary subgroup of an antiunitary group
        # use the antiunitary element A to generate all the elements G=H+AH
        A_spin = np.array([[1,0,0],[0,1,0],[0,0,-1]])
        A_rot = np.eye(3)
        A_tau = np.array([0,0,0])
        Time_R = -1j * np.array([[0, -1j], [1j, 0]])
        A_su2 = Time_R @ np.conj(SU2(-A_spin))
        # A_su2 = SU2(A_spin)
        for i,rot in enumerate(space_rot_uni):
            tau = space_tau_uni[i]
            spin = spin_uni[i]
            su2 =  su2_uni[i]
            space_rot_tot.append(A_rot @ rot)
            space_tau_tot.append(A_rot @ tau + A_tau)
            spin_tot.append(A_spin @ spin)
            su2_tot.append(A_su2 @ np.conj(su2))
            time_reversal.append(-1)
        self.rotC = space_rot_tot
        self.tauC = space_tau_tot
        self.spin = spin_tot
        self.su2s = su2_tot
        self.time_reversal = time_reversal

    def load_ssg_1d(self, ssg_dic):
        self.anti_spin = np.array([[-1,0,0],[0,1,0],[0,0,1]])
        self.anti_rotC = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.anti_tau = np.array([0,0,0])
        self.superCell = ssg_dic['superCell']
        self.Gid = ssg_dic['Gid']
        HRot = ssg_dic['HRotC']
        HTau = ssg_dic['HTauC']
        QRot = ssg_dic['QRotC']
        QTau = ssg_dic['QTauC']
        URot = ssg_dic['URot'][-1]
        # uni = uni_or_anti(ssg_dic)
        # assert uni[0] == -1
        # A_spin = uni[1]
        # A_rot = uni[2]
        # A_tau = uni[3]
        # generate a new element list
        space_rot_uni = []
        space_tau_uni = []
        spin_uni = []
        su2_uni = []
        space_rot_tot = []
        space_tau_tot = []
        spin_tot = []
        su2_tot = []
        time_reversal = []
        for i, hrot in enumerate(HRot):
            htau = HTau[i]
            for j, spin in enumerate(URot):
                if det(spin) > 0:
                    qrot = QRot[j]
                    qtau = QTau[j]
                    # r1 t1 * r2 t2 = r1r2| r1t2 + t1
                    # qrot|qtau * hrot|htau = qrot*hrot| qrot*htau + qtau
                    rot_new = qrot @ hrot
                    tau_new = qrot @ htau + qtau
                    su2_new = SU2(spin)
                else:
                    spin = np.array([[-1,0,0],[0,1,0],[0,0,1]]) @ spin
                    qrot = QRot[j]
                    qtau = QTau[j]
                    # r1 t1 * r2 t2 = r1r2| r1t2 + t1
                    # qrot|qtau * hrot|htau = qrot*hrot| qrot*htau + qtau
                    rot_new = qrot @ hrot
                    tau_new = qrot @ htau + qtau
                    su2_new = SU2(spin)
                spin_c2z = np.array([[-1,0,0],[0,-1,0],[0,0,1]]) @ spin
                su2_c2z = SU2(spin_c2z)
                space_rot_uni.append(rot_new)
                space_tau_uni.append(tau_new)
                spin_uni.append(spin)
                su2_uni.append(su2_new)
                time_reversal.append(1)
                space_rot_tot.append(rot_new)
                space_tau_tot.append(tau_new)
                spin_tot.append(spin)
                su2_tot.append(su2_new)
                # double the unitary part by C2z
                space_rot_uni.append(rot_new)
                space_tau_uni.append(tau_new)
                spin_uni.append(spin_c2z)
                su2_uni.append(su2_c2z)
                time_reversal.append(1)
                space_rot_tot.append(rot_new)
                space_tau_tot.append(tau_new)
                spin_tot.append(spin_c2z)
                su2_tot.append(su2_c2z)

        # here the space_rot, space_tau, spin, su2 are the unitary subgroup of an antiunitary group
        # use the antiunitary element A to generate all the elements G=H+AH
        A_spin = np.array([[-1,0,0],[0,1,0],[0,0,1]])
        A_rot = np.eye(3)
        A_tau = np.array([0,0,0])
        Time_R = -1j * np.array([[0, -1j], [1j, 0]])
        A_su2 = Time_R @ np.conj(SU2(-A_spin))
        # A_su2 = SU2(A_spin)
        for i,rot in enumerate(space_rot_uni):
            tau = space_tau_uni[i]
            spin = spin_uni[i]
            su2 =  su2_uni[i]
            space_rot_tot.append(A_rot @ rot)
            space_tau_tot.append(A_rot @ tau + A_tau)
            spin_tot.append(A_spin @ spin)
            su2_tot.append(A_su2 @ np.conj(su2))
            time_reversal.append(-1)
        self.rotC = space_rot_tot
        self.tauC = space_tau_tot
        self.spin = spin_tot
        self.su2s = su2_tot
        self.time_reversal = time_reversal

    def ssg_bz(self):
        def calculate_reciprocal_lattice(pure_t):
            # cal the reciprocal basis from pure_t
            a1, a2, a3 = pure_t
            V = np.dot(a1, np.cross(a2, a3))
            b1 = (2 * np.pi * np.cross(a2, a3)) / V
            b2 = (2 * np.pi * np.cross(a3, a1)) / V
            b3 = (2 * np.pi * np.cross(a1, a2)) / V
            
            return b1, b2, b3
        tau_col = self.superCell
        gid = self.Gid
        prim_vec = identify_SG_lattice(gid)[1] # each col is a prim basis vector   
        t1 = np.array([tau_col[0][0], tau_col[1][0], tau_col[2][0]]) 
        t2 = np.array([tau_col[0][1], tau_col[1][1], tau_col[2][1]]) 
        t3 = np.array([tau_col[0][2], tau_col[1][2], tau_col[2][2]])
        pure_t = []
        for t_append in [t1, t2, t3]:
            pure_t.append(prim_vec @ t_append)
        self.pure_T = pure_t
        # pure t form the a_{i,j,k} in the basis of conventional basis of G
        b1, b2, b3 = calculate_reciprocal_lattice(pure_t)
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def get_table(self):
        from ssg_numeric import multiplication_table
        self.mul_table, self.factor_su2 = multiplication_table(self)

    def get_trans_factor(self, kpoint):
        time_reversal = self.time_reversal
        degreee = len(time_reversal)
        space_rot = self.rotC
        space_tau = self.tauC
        factor = np.zeros((degreee, degreee),dtype=complex)
        b1, b2, b3 = self.b1, self.b2, self.b3
        k_conv = kpoint[0] * b1 + kpoint[1] * b2 + kpoint[2] * b3
        for i, rot1 in enumerate(space_rot):
            # rot1_k = inv(self.pure_T) @ rot1 @ self.pure_T
            # rot1_k = inv(rot1_k.T)
            tau1 = space_tau[i]
            t1 = time_reversal[i]
            for j, tau2 in enumerate(space_tau):
                f = np.exp( -1j * np.dot(k_conv, rot1 @ tau2 - t1 * tau2))
                # f = np.exp(- 1j * np.dot(t1 * inv(rot1)@ k_conv - k_conv, tau2))
                # factor[i][j] = f * np.exp(-1j * np.dot(k_conv, rot1 @ tau2 + tau1))
                factor[i][j] = f
        # print_mat(factor)
        self.factor_trans = factor


def loadSsgGroup(ssgNum, kvec, group_type, ssg_dic):
    ssg = ssgGroup(ssgNum, group_type)
    if 'P' in ssgNum:
        # print('coplanar condition')
        ssg.load_ssg_2d(ssg_dic)
    elif 'L' in ssgNum:
        ssg.load_ssg_1d(ssg_dic)
    else:
        ssg.load_ssg(ssg_dic)
    ssg.ssg_bz()
    ssg.get_table()
    ssg.get_trans_factor(kvec)
    return ssg


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


class LittleGroup:  # little group at a special k point
    def __init__(self, ssgNum, kvec, group_type, ssg_dic, need_beautify):
        self.ssg = loadSsgGroup(ssgNum, kvec, group_type, ssg_dic)
        self.need_beautify = need_beautify
        self.ssgNum =ssgNum  # ssg number
        self.kvec = kvec  # k vector
        self.antiunitary = False
        self.oplist = []
        self.anti_index = 0
        self.rotC = []  # the rotation part of all operations as a list
        self.tauC = []  # the translation vector of all operations as a list
        self.spin = []
        self.su2s = []
        self.time_reversal = []
        self.pure_T = []
        self.Dv = [] # representation on k
        self.eta_0 = 0
        self.hT_index = []
        self.Fh = []
        self.mul_table = []
        self.factor = []
        self.regularRep = []
        self.rep_num = 0
        self.character = []
        self.linear_character = []
        self.correp_matrix = []
        self.Wmatrix = []
        self.classes = []
        self.subgroup = []
    
    def to_dic_json(self):
        return{
            "ssgNum": self.ssgNum,
            "kvec": convert_to_serializable(self.kvec),
            "rotC": convert_to_serializable(self.rotC),
            "spin": convert_to_serializable(self.spin),
            "su2": convert_to_serializable(self.su2s),
            "timeReversal": convert_to_serializable(self.time_reversal),
            "tauC": convert_to_serializable(self.tauC),
            "character": convert_to_serializable(self.character),
            "repMatrix": convert_to_serializable(self.correp_matrix),
            "torsion": convert_to_serializable(self.torsion),
            "repDegree": convert_to_serializable(self.rep_degree),
            "matrixConvention": "projective",
            "characterConvention": "projective",
            "linearCharacter": convert_to_serializable(self.linear_character),
            "pureTranslations": convert_to_serializable(np.asarray(self.pure_T)),
            "reciprocalBasis": convert_to_serializable(np.asarray([self.b1, self.b2, self.b3])),
            "validation": self.validation
        }

    def to_dic_npy(self):
        return{
            "ssgNum": self.ssgNum,
            "kvec": self.kvec,
            "rotC": self.rotC,
            "spin": self.spin,
            "su2": self.su2s,
            "timeReversal": self.time_reversal,
            "tauC": self.tauC,
            "character": self.character,
            "repMatrix": self.correp_matrix,
            "torsion": self.torsion,
            "repDegree": self.rep_degree,
            "matrixConvention": "projective",
            "characterConvention": "projective",
            "linearCharacter": self.linear_character,
            "pureTranslations": np.asarray(self.pure_T),
            "reciprocalBasis": np.asarray([self.b1, self.b2, self.b3]),
            "validation": self.validation
        }
    
    def k_op1(self):
        def identity_kpoint(kpoint, r, t, pure_t):
            R_supercell = pure_t @ r @ inv(pure_t)
            # print(R_supercell)
            k_new = t * (inv(R_supercell)).T @ kpoint
            # k_new = t * R_supercell @ kpoint
            check = kpoint - k_new
            if norm(check- np.round(check)) < 1e-4:
                return True
            else:
                return False
        # a1, a2, a3 = get_pure_translation(ssg)
        # P = np.array([a1, a2, a3])
        ssg = self.ssg
        pure_t = np.array(ssg.pure_T)
        # print(pure_t)
        rot = ssg.rotC
        time_reversal = ssg.time_reversal
        self.b1, self.b2, self.b3 = ssg.b1, ssg.b2, ssg.b3
        kop_list = []
        for i, r in enumerate(rot):
            # print_mat(r)
            t= time_reversal[i]
            if identity_kpoint(self.kvec, r, t, pure_t):
                kop_list.append(i)
                if t < 0:
                    self.antiunitary = True
        self.oplist = kop_list
        # print(kop_list)
    
    def k_op(self):
        def identity_kpoint(kpoint, r, t, pure_t):
            t1, t2, t3 = pure_t
            # R_supercell = pure_t @ r @ inv(pure_t)
            # print(R_supercell)
            # print_mat(t * (inv(R_supercell.T)))
            # k_new = t * (inv(R_supercell)).T @ kpoint
            # k_new = t * R_supercell @ kpoint
            for tau in pure_t:
                factor = 1 - np.exp( 1j * np.dot(k_conv, -tau + t * r @ tau))
                if abs(factor) > 1e-8:
                    return False
            return True
            # check = kpoint - k_new
            # print(check)
            # if norm(check- np.round(check)) < 1e-4:
            #     return True
            # else:
            #     return False
        # a1, a2, a3 = get_pure_translation(ssg)
        # P = np.array([a1, a2, a3])
        ssg = self.ssg
        pure_t = np.array(ssg.pure_T)
        # print(pure_t)
        rot = ssg.rotC
        time_reversal = ssg.time_reversal
        self.b1, self.b2, self.b3 = ssg.b1, ssg.b2, ssg.b3
        b1, b2, b3 = self.b1, self.b2, self.b3
        kpoint = self.kvec
        k_conv = kpoint[0] * b1 + kpoint[1] * b2 + kpoint[2] * b3
        kop_list = []
        for i, r in enumerate(rot):
            # print_mat(r)
            t= time_reversal[i]
            if identity_kpoint(k_conv, r, t, pure_t):
                kop_list.append(i)
                if t < 0:
                    self.antiunitary = True
        # kop_list = [0, 1, 2, 3]
        self.antiunitary = any(time_reversal[i] < 0 for i in kop_list)
        self.oplist = kop_list
        # print(kop_list)
    
    def get_regular_rep_list(self, build_dense=True):
        def get_mul_table(table, op_list):
            # print(table)
            # print(op_list)
            row = np.size(table, 0)
            col = np.size(table, 1)
            new_table = np.zeros((row, col))
            assert row == col
            for i in range(row):
                for j in range(col):
                    ele = int(table[i][j])
                    new_ele = op_list.index(ele - 1) + 1
                    new_table[i][j] = new_ele
            return new_table
        #
        def regular_rep_from_table(mul_table, factor_table, m):
            degree = np.size(mul_table, 0)
            rep = np.zeros((degree, degree), dtype=complex)
            for i in range(degree):
                mul = mul_table[m][i]
                factor = factor_table[m][i]
                # if mul > 0:
                #     double = 1
                # else:
                #     double = -1
                # mul = double * mul
                mul = mul - 1
                assert abs(mul-round(mul)) < 1e-4
                mul = round(mul)
                rep[mul][i] = factor
            return rep
        rep_list = []
        k_op_list = self.oplist
        ssg = self.ssg
        time_reversal_all = ssg.time_reversal
        time_reversal = [time_reversal_all[i] for i in k_op_list]
        # degree = len(k_op_list)
        new_mul_table = extract_submatrix(ssg.mul_table, k_op_list)
        new_mul_table = get_mul_table(new_mul_table, k_op_list)
        new_factor_table = extract_submatrix(ssg.factor_su2, k_op_list)
        new_trans_factor = extract_submatrix(ssg.factor_trans, k_op_list)
        self.mul_table = new_mul_table.astype(int)
        self.factor = np.multiply(new_factor_table, new_trans_factor)
        if build_dense:
            for num, op_num in enumerate(k_op_list):
                rep_list.append(regular_rep_from_table(new_mul_table, self.factor, num))
        # return rep_list, time_reversal
        self.regularRep = rep_list
        self.time_reversal = time_reversal
        self.pure_T = ssg.pure_T
        self.rotC = [ssg.rotC[i] for i in k_op_list]
        self.tauC = [ssg.tauC[i] for i in k_op_list]
        self.spin = [ssg.spin[i] for i in k_op_list]
        self.su2s = [ssg.su2s[i] for i in k_op_list]


    def get_irreducible_rep(self):
        from ssg_numeric import reduce_regular
        reduce_regular(self, seed=getattr(self, 'seed', 0))

    def beautify_rep_subgroups(self, repnum):
        """Compatibility wrapper for a safe unitary basis optimization."""
        from ssg_numeric import sparse_basis
        return sparse_basis(self.correp_matrix[repnum], self.time_reversal)

    def solve_factor(self, repnum):
        """Compatibility no-op: independent entry rescaling is not a basis change."""
        return np.array(self.correp_matrix[repnum], copy=True)

    def cal_classes(self):
        def find_inversion(element, multable):
            row = multable[element]
            ind = np.argwhere(row == 1)
            assert len(ind) == 1
            return ind[0][0]
        def find_one_class(element, multable, time_reversal): # element is an element number from 0
            degree = np.size(multable, 0)
            class1 = [element]
            for i in range(degree):
                element_inv = find_inversion(i, multable)
                if time_reversal[i] > 0:
                    mul1 = multable[element_inv][element] - 1
                    mul2 = multable[mul1][i] - 1
                else:
                    h_inv = find_inversion(element, multable)
                    mul1 = multable[element_inv][h_inv] - 1
                    mul2 = multable[mul1][i] - 1
                if mul2 not in class1:
                    class1.append(mul2)
            return class1
        def find_calsses(multable, time_reversal):
            def exit_class(element, class_list):
                for class1 in class_list:
                    if element in class1:
                        return True
                return False
            class_list = [[0]]
            degree = np.size(multable, 0)
            if -1 in time_reversal:
                uni_degree = degree//2
            else:
                uni_degree = degree
            for i in range(uni_degree):
                if not exit_class(i, class_list):
                    class_list.append(find_one_class(i, multable, time_reversal))
            d = 0
            for class1 in class_list:
                d = d + len(class1)
            assert abs(d  - uni_degree) < 1e-3
            return class_list
        mul_table = self.mul_table
        # unitary_degree = len(self.time_reversal)//2
        time_reversal = self.time_reversal
        # unitary_mul_table = extract_submatrix(mul_table, list(range(unitary_degree)))
        classes = find_calsses(mul_table, time_reversal)
        self.classes = classes
        return classes


    def regular_subgroup(self):
        def check_subgroup(mul_table, checkarray):
            sub_multable = extract_submatrix(mul_table, checkarray)
            len1 = len(checkarray)
            len2 = np.size(mul_table, 0)
            qlen = len2/len1
            if abs(qlen - round(qlen)) > 1e-3:
                return False
            for row in sub_multable:
                for ele in row:
                    ele = ele -1
                    if ele not in checkarray:
                        return False
            return True
        def class2subgroup(classes, mul_table):
            subgroup = []
            len1 = np.size(mul_table, 0)
            num_classes = len(classes)
            num_loop = num_classes - 2
            not_special = True
            if len(classes) == len1//2:
                not_special = False
            while num_loop > 0:
                # print(num_loop)
                if not_special or num_loop == 1: 
                    sub_classes_list = list(combinations(classes[1:], num_loop))
                    for sub_classes in sub_classes_list:
                        sub_classes = combine_array(sub_classes)
                        sub_classes.append(0)
                        if check_subgroup(mul_table, sub_classes):
                            subgroup.append(sub_classes)
                num_loop = num_loop - 1
            return subgroup
        classes = self.cal_classes()
        mul_table = self.mul_table
        subgroups = class2subgroup(classes, mul_table)
        def sort_key(arr):
            return len(arr)
        sorted_list = sorted(subgroups, key=sort_key, reverse=True)
        sorted_list.append([0])
        self.subgroup = sorted_list
        return sorted_list



    
    def get_kdotp(self):
        dv = []
        for i, r in enumerate(self.rotC):
            t = self.time_reversal[i]
            pure_t = self.pure_T
            r_k = inv(pure_t) @ r @ pure_t
            # print('det', det(r_k))
            dv.append(t * inv(r_k.T))
        self.Dv = dv
        degree = len(self.time_reversal)
        anti_index = degree // 2
        self.anti_index = anti_index
        assert self.time_reversal[degree//2] == -1
        self.eta_0 = self.factor[anti_index, anti_index]
        self.hT_index = self.mul_table[0: anti_index, anti_index]
        Fh = []
        rep_num = np.size(self.character,0)
        # print(rep_num)
        for rep in range(rep_num):
            fhh = []
            correp = self.correp_matrix[rep]
            for uni in range(anti_index):
                fffh=np.dot(np.dot(correp[anti_index], np.conj(correp[uni])), np.conj(correp[anti_index].T))
                fhh.append(fffh)
            Fh.append(fhh)
        self.Fh = Fh

    def get_W(self, rep_num):# rep_num being the number of representation
        Wmatrix = []
        degree = len(self.time_reversal)
        anti_index = degree // 2
        Dv = self.Dv
        correp = self.correp_matrix[rep_num]
        # print_matlist(correp)
        Fh = self.Fh[rep_num]
        for uni in range(anti_index):
            # degreeW = np.size(Dv[0], 0) * np.size(correp[0], 0) * np.size(Fh[0], 0)
            # part1 = np.zeros((degreeW, degreeW))
            part1 = np.kron(Dv[uni], np.kron(correp[uni], Fh[uni]))
            FdotM = np.dot(Fh[uni], correp[self.mul_table[anti_index, anti_index]-1])# -1 because the multible table index is from 1 
            MFM = TwistOperator(correp[uni], FdotM)
            part2 = np.kron(Dv[self.hT_index[uni]-1], MFM) # -1 because the multible table index is from 1 
            Wmatrix.append(0.5*(part1 + self.eta_0 * part2))
        # print_matlist(Wmatrix)
        self.Wmatrix = Wmatrix
            # print_mat(part1）
        
    # def get_multiplicity(self, rep_num):
        degree = len(self.time_reversal)
        anti_index = degree // 2
        Dv = self.Dv
        correp = self.correp_matrix[rep_num]
        ai = 0
        for uni in range(anti_index):
            character_uni = np.trace(correp[uni])
            part1 = abs(character_uni)* abs(character_uni) *np.trace(Dv[uni])
            index_hT = self.hT_index[uni] - 1
            omega = self.factor[index_hT, index_hT]
            kappa = correp[self.mul_table[index_hT, index_hT] -1]
            part2 = np.trace(Dv[index_hT])* omega * np.trace(kappa)
            # print(part1)
            # print(part2)
            ai = ai + (part1 + part2)/(degree)
        aii = round(abs(ai))
        assert norm(aii - ai)< 1e-5
        # print('the number of independent matrices from formula:')
        # print(aii)
        # return aii
    
    # def common_null_space(self):
        # eigenvalues_B, eigenvectors_B = np.linalg.eig(Wmatrix[0])
        # print(eigenvalues_B)
        # Wmatrix = self.Wmatrix
        Sn = []
        for mat in Wmatrix:
            ssn = mat - np.eye(np.size(mat,0))
            Sn.append(ssn)
        # print_matlist(Sn)
        Ssn = np.vstack(Sn)  
        nullspace = null_space(Ssn)
        # print('the number of independent matrices from solving nullspace:')
        # print(np.size(nullspace, 1))
        # change the bases WT:
        anti_index = self.anti_index
        mt0 = correp[anti_index]
        WT = np.kron(self.Dv[anti_index], np.kron(mt0, mt0))
        TInSub = np.conj(nullspace.T) @ WT @ np.conj(nullspace)
        rpt = sqrtm(TInSub)
        # print(np.conj(rpt.T) @ TInSub @ np.conj(rpt))
        BasisW = nullspace @ rpt
        dim_dv = np.size(Dv[0], 0)
        dim_rep = np.size(mt0, 0)
        assert aii == np.size(nullspace, 1)
        for i in range(aii):
            print(i+1, 'th independent kdotp Hamiltonian:')
            for kd in range(dim_dv):
                ll = BasisW[kd * dim_rep* dim_rep : kd * dim_rep* dim_rep + dim_rep* dim_rep, i]
                gamma = np.reshape(ll, (dim_rep, dim_rep))
                # numpy reshape 1: 4 to 2,2: [[1,2],[3.4]]
                # matlab reshape " [[1,3],[2,4]]"
                Gamma_final = gamma @ np.conj(mt0)
                kkp = ['kx', 'ky', 'kz']
                print(kkp[kd], ':')
                print_mat(Gamma_final)
        
        # check = Ssn @ solve1
        # print(norm(check))




def load_little_group(ssgNum, kvec, need_beautify, group_type, ssg_dic, seed=0):
    lg = LittleGroup(ssgNum, kvec, group_type, ssg_dic, need_beautify)
    lg.seed = seed
    lg.k_op()
    lg.get_regular_rep_list(build_dense=False)
    lg.get_irreducible_rep()
    # lg.get_kdotp()
    # lg.get_W(1)
    return lg

# def common_null_space(Wmatrix):
#     # eigenvalues_B, eigenvectors_B = np.linalg.eig(Wmatrix[0])
#     # print(eigenvalues_B)
#     Sn = []
#     for mat in Wmatrix:
#         ssn = mat - np.eye(np.size(mat,0))
#         Sn.append(ssn)
#     # print_matlist(Sn)
#     Ssn = np.vstack(Sn)  
#     nullspace = null_space(Ssn, 1e-2)
#     print('the number of independent matrices from solving nullspace:')
#     print(np.size(nullspace, 1))
#     solve1 = nullspace[:,0]
#     check = Ssn @ solve1
#     print(norm(check))

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)

def load_one_ssg_kvec(ssgnum, kvec, single=2, out='character', fileType=None,
                     optimize=False, convention='bloch', rep=None, ops=None,
                     compact_characters=False, seed=0, html_path=None,
                     text_path=None, export_path=None):
    """Compute, validate, and report one SSG little group.

    kvec uses the reciprocal basis of pure translations. JSON/NPY matrices
    remain projective; the selected display convention applies to reports.
    """
    from pathlib import Path
    import contextlib
    from ssg_database import load_database
    from ssg_output import from_little_group, report_text, report_html
    kvec = np.asarray(kvec, dtype=float)
    if kvec.shape != (3,) or not np.isfinite(kvec).all():
        raise ValueError('k must contain three finite real coordinates.')
    if single not in (1, 2):
        raise ValueError('Group type must be 1 (single-valued) or 2 (double-valued).')
    if out not in ('character', 'rep_matrix', 'rep_degree', 'both'):
        raise ValueError('Unknown output type.')
    if fileType not in (None, 'json', 'npy'):
        raise ValueError('Export type must be json or npy.')
    if convention not in ('bloch', 'projective'):
        raise ValueError('Unknown display convention.')
    if not isinstance(optimize, (bool, np.bool_)):
        raise ValueError('Optimize must be a boolean.')
    if rep is not None and rep < 1:
        raise ValueError('Representation indices start at 1.')
    if ops is not None and (not ops or min(ops) < 1):
        raise ValueError('Operation indices start at 1.')
    if export_path is not None and fileType is None:
        raise ValueError('--export requires --fileType json or npy.')
    database = load_database()
    if ssgnum not in database:
        raise ValueError('Unknown SSG number: '+ssgnum)
    with contextlib.redirect_stdout(sys.stderr):
        lg = load_little_group(ssgnum, kvec, optimize,
                              'single' if single == 1 else 'double', database[ssgnum], seed=seed)
    if rep is not None and rep > lg.rep_num:
        raise ValueError('Representation index exceeds the number of corepresentations.')
    if ops is not None and max(ops) > len(lg.time_reversal):
        raise ValueError('Operation index exceeds this little group order.')
    data = from_little_group(lg)
    text = report_text(data, rep_label='r'+str(rep) if rep is not None else None,
                       operation_ids=ops, convention=convention, output=out,
                       compress=compact_characters)
    def write_text(path, content):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    if html_path:
        chosen_rep = rep if rep is not None else int(np.argmax(lg.rep_degree))+1
        chosen_ops = ops
        if chosen_ops is None:
            chosen_ops = list(range(1, min(4, len(lg.time_reversal))+1))
            anti = [i+1 for i, t in enumerate(lg.time_reversal) if t < 0]
            if anti and anti[0] not in chosen_ops:
                chosen_ops.append(anti[0])
        write_text(html_path, report_html(data, 'r'+str(chosen_rep), chosen_ops,
                                         convention, compress=compact_characters))
    if text_path:
        write_text(text_path, text)
    if fileType:
        destination = Path(export_path or 'output.'+fileType)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if fileType == 'json':
            destination.write_text(json.dumps(lg.to_dic_json(), cls=CustomEncoder, indent=2), encoding='utf-8')
        else:
            with destination.open('wb') as stream:
                np.save(stream, lg.to_dic_npy())
    print(text, end='')
    return lg


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false.')


def main():
    parser = argparse.ArgumentParser(description='Compute and validate SSG little-group corepresentations.')
    parser.add_argument('--ssgNum', required=True, help='SSG database identifier')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--kp', nargs=3, type=float, metavar=('kx', 'ky', 'kz'),
                        help='Numerical coordinates in the reciprocal basis of pure translations')
    inputs.add_argument('--kpath', '--kp-symbolic', nargs=3, metavar=('kx(t)', 'ky(t)', 'kz(t)'),
                        help='Symbolic path coordinates, for example: "1/2" "1/2" "t"')
    parser.add_argument('--parameter', help='Real path parameter name; default: t (requires --kpath)')
    parser.add_argument('--interval', nargs=2, metavar=('LOWER', 'UPPER'),
                        help='Open path interval with exact bounds; default: 0 1 (requires --kpath)')
    parser.add_argument('--groupType', type=int, choices=[1, 2], default=2)
    parser.add_argument('--out', choices=['character', 'rep_matrix', 'rep_degree', 'both'], default='character')
    parser.add_argument('--fileType', choices=['json', 'npy'])
    parser.add_argument('--export', help='Export path; requires --fileType')
    parser.add_argument('--optimize', type=parse_bool, nargs='?', const=True, default=False,
                        help='Use a safe unitary basis optimization (true/false)')
    parser.add_argument('--seed', type=int, default=0, help='Local random seed for reproducible reduction')
    parser.add_argument('--convention', choices=['bloch', 'projective'], default='bloch')
    parser.add_argument('--rep', type=int, help='One-based representation index for matrices')
    parser.add_argument('--ops', type=int, nargs='+', help='One-based operation indices within this little group')
    parser.add_argument('--compact-characters', action='store_true',
                        help='Merge exactly equal character vectors, not conjugacy classes')
    parser.add_argument('--html', help='Write an offline HTML report (default: largest representation)')
    parser.add_argument('--text', help='Also save the aligned stdout report to this file')
    args = parser.parse_args()
    try:
        if args.kpath is not None:
            from ssg_symbolic import load_one_ssg_path
            load_one_ssg_path(args.ssgNum, args.kpath, args.parameter or 't', args.interval or ('0', '1'),
                              args.groupType, args.out, args.fileType, args.optimize, args.convention,
                              args.rep, args.ops, args.compact_characters, args.seed,
                              args.html, args.text, args.export)
        else:
            if args.parameter is not None or args.interval is not None:
                raise ValueError('--parameter and --interval require --kpath.')
            load_one_ssg_kvec(args.ssgNum, args.kp, args.groupType, args.out, args.fileType,
                             args.optimize, args.convention, args.rep, args.ops,
                             args.compact_characters, args.seed, args.html, args.text, args.export)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, 'Error: '+str(exc)+'\n')


if __name__ == '__main__':
    main()
