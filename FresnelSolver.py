import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

class layerClass():
    def __init__(self, t, eps, s):
        self.t = t
        self.eps = eps
        self.s = s        
        self.Sc = [[[],[]],[[],[]]]

class interface():
    def __init__(self, SParams_S = None, SParams_P = None):
        self.ss = SParams_S
        self.sp = SParams_P
        self.default = False
        if SParams_S is None:
            self.default = True            
        self.Sc = [[[],[]],[[],[]]]

def main():
    freqs = np.linspace(200, 1700, 1000) * 1e9 #0.2 to 1.7 THz

    layers = [layerClass(500e-6, 3**2, 0), layerClass(100e-6, 1.5**2, 0), layerClass(100e-6, 1**2, 300)]
    interfaces = [interface(), interface(), interface(), interface()]

    theta = 0
    phi = 0

    SMat, SMat4x4 = fresnelSolv(freqs, theta, phi, layers, interfaces, 1)

    S_sMatTheory =    [SMat[0][0], SMat[2][0], SMat[1][0], SMat[3][0]]
    S_pMatTheory =    [SMat[0][1], SMat[2][1], SMat[1][1], SMat[3][1]]
    S_sMatRevTheory = [SMat[2][2], SMat[0][2], SMat[3][2], SMat[1][2]]
    S_pMatRevTheory = [SMat[2][3], SMat[0][3], SMat[3][3], SMat[1][3]]

    
    plotfreqs = freqs * 1e-12    
    #Plot forward parallel
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig.suptitle(r'$Scattering\/Parameters\/(S_{forward,\parallel})$')
    
    line1, = ax1.plot(plotfreqs, np.absolute(S_pMatTheory[0]), 'r')
    ax1.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax1.set_title(r'$\mid R_{\perp\parallel} \mid$') 
    
    line1, = ax2.plot(plotfreqs, np.absolute(S_pMatTheory[1]), 'r')
    ax2.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax2.set_title(r'$\mid T_{\perp\parallel} \mid$') 
    
    line1, = ax3.plot(plotfreqs, np.absolute(S_pMatTheory[2]), 'r')
    ax3.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax3.set_title(r'$\mid R_{\parallel \parallel}\mid$') 
    ax3.set_xlabel('Freq (THz)')
    
    line1, = ax4.plot(plotfreqs, np.absolute(S_pMatTheory[3]), 'r')
    ax4.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax4.set_title(r'$\mid T_{\parallel \parallel}\mid$') 
    ax4.set_xlabel('Freq (THz)')
    
    #Plot forward perp
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig.suptitle(r'$Scattering\/Parameters\/(S_{forward,\perp})$')
    
    line1, = ax1.plot(plotfreqs, np.absolute(S_sMatTheory[0]), 'r')
    ax1.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax1.set_title(r'$\mid R_{\perp\perp} \mid$') 
    
    line1, = ax2.plot(plotfreqs, np.absolute(S_sMatTheory[1]), 'r')
    ax2.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax2.set_title(r'$\mid T_{\perp\perp} \mid$') 
    
    line1, = ax3.plot(plotfreqs, np.absolute(S_sMatTheory[2]), 'r')
    ax3.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax3.set_title(r'$\mid R_{\parallel \perp}\mid$') 
    ax3.set_xlabel('Freq (THz)')
    
    line1, = ax4.plot(plotfreqs, np.absolute(S_sMatTheory[3]), 'r')
    ax4.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax4.set_title(r'$\mid T_{\parallel \perp}\mid$') 
    ax4.set_xlabel('Freq (THz)')
    
    #Plot back parallel
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig.suptitle(r'$Scattering\/Parameters\/(S_{backward,\parallel})$')
    
    line1, = ax1.plot(plotfreqs, np.absolute(S_pMatRevTheory[0]), 'r')
    ax1.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax1.set_title(r'$\mid R_{\perp\parallel} \mid$') 
    
    line1, = ax2.plot(plotfreqs, np.absolute(S_pMatRevTheory[1]), 'r')
    ax2.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax2.set_title(r'$\mid T_{\perp\parallel} \mid$') 
    
    line1, = ax3.plot(plotfreqs, np.absolute(S_pMatRevTheory[2]), 'r')
    ax3.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax3.set_title(r'$\mid R_{\parallel \parallel}\mid$') 
    ax3.set_xlabel('Freq (THz)')
    
    line1, = ax4.plot(plotfreqs, np.absolute(S_pMatRevTheory[3]), 'r')
    ax4.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax4.set_title(r'$\mid T_{\parallel \parallel}\mid$') 
    ax4.set_xlabel('Freq (THz)')
    
    #Plot back perp
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    fig.suptitle(r'$Scattering\/Parameters\/(S_{backward,\perp})$')
    
    line1, = ax1.plot(plotfreqs, np.absolute(S_sMatRevTheory[0]), 'r')
    ax1.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax1.set_title(r'$\mid R_{\perp\perp} \mid$') 
    
    line1, = ax2.plot(plotfreqs, np.absolute(S_sMatRevTheory[1]), 'r')
    ax2.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax2.set_title(r'$\mid T_{\perp\perp} \mid$') 
    
    line1, = ax3.plot(plotfreqs, np.absolute(S_sMatRevTheory[2]), 'r')
    ax3.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax3.set_title(r'$\mid R_{\parallel \perp}\mid$') 
    ax3.set_xlabel('Freq (THz)')
    
    line1, = ax4.plot(plotfreqs, np.absolute(S_sMatRevTheory[3]), 'r')
    ax4.axis([min(plotfreqs), max(plotfreqs), 0, 1])
    ax4.set_title(r'$\mid T_{\parallel \perp}\mid$') 
    ax4.set_xlabel('Freq (THz)')
    
    plt.show()


def redhefferstar(SA11, SA12, SA21, SA22, SB11, SB12, SB21, SB22):

    # redhefferstar product combines two scattering matrices to form a overall
    # scattering matrix. It is used in forming scattering matrices of
    # dielectric stacks in transfer matrix method
    # SAXX and SBXX are scattering matrices of two different layers
    # and this function outputs the combined scaterring matrix of two layers
    I = np.eye(len(SA11))
    
    SAB11 = SA11+(SA12*(np.linalg.inv(I-(SB11*SA22)))*SB11*SA21)
    SAB12 = SA12*(np.linalg.inv(I-(SB11*SA22)))*SB12
    SAB21 = SB21*(np.linalg.inv(I-(SA22*SB11)))*SA21
    SAB22 = SB22+(SB21*(np.linalg.inv(I-(SA22*SB11)))*SA22*SB12)
    return [SAB11, SAB12, SAB21, SAB22]


def fresnelSolv(freqs, tInc, pInc, alayers, interfaces, nBackground):
    
    c = 299792458
    e0 = 8.854187817e-12
    
    totLayers = [layerClass(-1, nBackground**2, 0)]
    totLayers[1:] = alayers
    totLayers.append(layerClass(-1, nBackground**2, 0))

    ky = np.sin(tInc) * np.cos(pInc)
    kx = np.sin(tInc) * np.sin(pInc)

    SMat = [[0 for x in range(4)] for x in range(4)]
    SMat4x4 = []
    
    for i in range(len(SMat)):
        for j in range(len(SMat)):
            SMat[i][j] = []

    #Process the interfaces
    for intIndex, interface in enumerate(interfaces):
        if interface.default is False:                
            for fIndex, freq in enumerate(freqs):
                S_s = interface.ss[fIndex] #[Rss, Tss, Rps, Tps]
                S_p = interface.sp[fIndex] #[Rsp, Tsp, Rpp, Tpp]

                Rss = S_s[1]; Tss = S_s[2]; Rps = S_s[3]; Tps = S_s[4]
                Rsp = S_p[1]; Tsp = S_p[2]; Rpp = S_p[3]; Tpp = S_p[4]         

                interfaces[intIndex].Sc[0][0].append(np.matrix([[Rss, Rsp], [Rps, Rpp]]))
                interfaces[intIndex].Sc[0][1].append(np.matrix([[Tss, Tsp], [Tps, Tpp]]))
                interfaces[intIndex].Sc[1][0].append(np.matrix([[Tss, Tsp], [Tps, Tpp]]))
                interfaces[intIndex].Sc[1][1].append(np.matrix([[Rss, Rsp], [Rps, Rpp]]))

    #Process the layers
    e_h = 1
    P_h = (1 / e_h) * np.matrix([[kx * ky, e_h - kx**2], [ky**2 - e_h, -kx * ky]])
    Q_h = (1 / 1)   * np.matrix([[kx * ky, e_h - kx**2], [ky**2 - e_h, -kx * ky]])
    lam2_ht, W_h = np.linalg.eig(P_h * Q_h)
    lam2_h = np.diag(lam2_ht)
    lam_h = np.lib.scimath.sqrt(lam2_h)
    V_h = Q_h * W_h * np.linalg.inv(lam_h)
                
    for layerIndex, layer in enumerate(totLayers[1:-1]):
        for fIndex, freq in enumerate(freqs):
            k0 = (2 * np.pi * freq / c)
            e = layer.eps + 1j * layer.s / (e0 * 2 * np.pi * freq)
            
            P = (1 / e) * np.matrix([[kx * ky, e - kx**2], [ky**2 - e, -kx * ky]])
            Q = np.matrix([[kx * ky, e - kx**2], [ky**2 - e, -kx * ky]])
            lam2t, W = np.linalg.eig(P * Q)
            lam2 = np.diag(lam2t)
            lam = np.lib.scimath.sqrt(lam2)
            V = Q * W * np.linalg.inv(lam)
            X = linalg.expm(lam * k0 * layer.t)
            
            A = np.linalg.inv(W) * W_h + np.linalg.inv(V) * V_h
            B = np.linalg.inv(W) * W_h - np.linalg.inv(V) * V_h
            
            totLayers[1 + layerIndex].Sc[0][0].append(np.linalg.inv(A - X * B * np.linalg.inv(A) * X * B) * (X * B * np.linalg.inv(A) * X * A - B))
            totLayers[1 + layerIndex].Sc[0][1].append(np.linalg.inv(A - X * B * np.linalg.inv(A) * X * B) * X * (A - B * np.linalg.inv(A) * B))
            totLayers[1 + layerIndex].Sc[1][0].append(np.linalg.inv(A - X * B * np.linalg.inv(A) * X * B) * X * (A - B * np.linalg.inv(A) * B))
            totLayers[1 + layerIndex].Sc[1][1].append(np.linalg.inv(A - X * B * np.linalg.inv(A) * X * B) * (X * B * np.linalg.inv(A) * X * A - B))

    #Combine the scattering matrices
    for fIndex, freq in enumerate(freqs):
        
        ScTot = [[[],[]],[[],[]]]
        
        ScTot[0][0] = totLayers[1].Sc[0][0][fIndex]
        ScTot[0][1] = totLayers[1].Sc[0][1][fIndex]
        ScTot[1][0] = totLayers[1].Sc[1][0][fIndex]
        ScTot[1][1] = totLayers[1].Sc[1][1][fIndex]
        
        interface = interfaces[0]
        if interface.default is False:
            ScTot[0][0], ScTot[0][1], ScTot[1][0], ScTot[1][1] = redhefferstar(interface.Sc[0][0][fIndex], interface.Sc[0][1][fIndex], interface.Sc[1][0][fIndex], interface.Sc[1][1][fIndex], ScTot[0][0], ScTot[0][1], ScTot[1][0], ScTot[1][1])
            
        for layerIndex, layer in enumerate(totLayers[2:-1]):
            interface = interfaces[layerIndex + 1]
            
            if interface.default is False:
                ScTot[0][0], ScTot[0][1], ScTot[1][0], ScTot[1][1] = redhefferstar(ScTot[0][0], ScTot[0][1], ScTot[1][0], ScTot[1][1], interface.Sc[0][0][fIndex], interface.Sc[0][1][fIndex], interface.Sc[1][0][fIndex], interface.Sc[1][1][fIndex])
            
            ScTot[0][0], ScTot[0][1], ScTot[1][0], ScTot[1][1] = redhefferstar(ScTot[0][0], ScTot[0][1], ScTot[1][0], ScTot[1][1], layer.Sc[0][0][fIndex], layer.Sc[0][1][fIndex], layer.Sc[1][0][fIndex], layer.Sc[1][1][fIndex])
        
        SMat4x4.append(np.bmat([[ScTot[0][0], ScTot[0][1]], [ScTot[1][0], ScTot[1][1]]]))
        
        SMatSingle = SMat4x4[fIndex]
        for i in range(len(SMat)):
            for j in range(len(SMat)):
                SMat[i][j].append(SMatSingle[i, j])
   
    return [SMat, SMat4x4]
        

main()


