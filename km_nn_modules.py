import torch
import torch.nn as nn
import torch.nn.functional as F

# LinReg
class LinReg(nn.Module):
   def __init__(self):
       super(LinReg, self).__init__()
       self.layer = torch.nn.Linear(38, 1)

   def forward(self, x):
       x = self.layer(x)
       return x

# Multi Layer Perceptron
class MLPReg1(nn.Module):
    def __init__(self,DO):
        super(MLPReg1, self).__init__()
        self.layer1 = torch.nn.Linear(38, 152)
        #self.dropout1 = nn.Dropout(p=drop1, inplace=True)
        self.layer2 = torch.nn.Linear(152, 152)
        #self.dropout2 = nn.Dropout(p=drop2, inplace=True)
        self.layer3 = torch.nn.Linear(152, 76)
        #self.dropout3 = nn.Dropout(p=drop3, inplace=True)
        # self.layer4 = torch.nn.Linear(456, 228)
        # self.dropout4 = nn.Dropout(p=drop4, inplace=True)
        # self.layer5 = torch.nn.Linear(228, 114)
        # self.dropout5 = nn.Dropout(p=drop5, inplace=True)
        # self.layer6 = torch.nn.Linear(114, 38)
        # self.dropout6 = nn.Dropout(p=drop6, inplace=True)
        # self.layer7 = torch.nn.Linear(38, 19)
        # self.dropout7 = nn.Dropout(p=drop7, inplace=True)
        # self.layer8 = torch.nn.Linear(19, 10)
        # self.dropout8 = nn.Dropout(p=drop8, inplace=True)
        # self.layer9 = torch.nn.Linear(10, 5)
        # self.dropout9 = nn.Dropout(p=drop9, inplace=True)
        # self.layer10 = torch.nn.Linear(5, 3)
        # self.dropout10 = nn.Dropout(p=drop10, inplace=True)
        self.layer11 = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        #x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        #x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        #x = self.dropout3(x)
        # x = F.relu(self.layer4(x))
        # x = self.dropout4(x)
        # x = F.relu(self.layer5(x))
        # x = self.dropout5(x)
        # x = F.relu(self.layer6(x))
        # x = self.dropout6(x)
        # x = F.relu(self.layer7(x))
        # x = self.dropout7(x)
        # x = F.relu(self.layer8(x))
        # x = self.dropout8(x)
        # x = F.relu(self.layer9(x))
        # x = self.dropout9(x)
        # x = F.relu(self.layer10(x))
        # x = self.dropout10(x)
        x = self.layer11(x)

        return x

class MLPReg2(nn.Module):
    def __init__(self, DO):
        super(MLPReg2, self).__init__()
        self.layer1 = torch.nn.Linear(38, 152)
        #self.dropout1 = nn.Dropout(p=0.5, inplace=True)
        self.layer2 = torch.nn.Linear(152, 256)
        #self.dropout2 = nn.Dropout(p=0.5, inplace=True)
        self.layer3 = torch.nn.Linear(256, 256)
        #self.dropout3 = nn.Dropout(p=0.5, inplace=True)
        self.layer4 = torch.nn.Linear(256, 512)
        #self.dropout4 = nn.Dropout(p=0.5, inplace=True)
        self.layer5 = torch.nn.Linear(512, 512)
        #self.dropout5 = nn.Dropout(p=0.5, inplace=True)
        self.layer6 = torch.nn.Linear(512, 1024)
        #self.dropout6 = nn.Dropout(p=0.5, inplace=True)
        self.layer7 = torch.nn.Linear(1024, 1024)
        #self.dropout7 = nn.Dropout(p=0.5, inplace=True)
        self.layer8 = torch.nn.Linear(1024, 512)
        #self.dropout8 = nn.Dropout(p=0.5, inplace=True)
        self.layer9 = torch.nn.Linear(512, 256)
        #self.dropout9 = nn.Dropout(p=0.5, inplace=True)
        self.layer10 = torch.nn.Linear(256, 128)
        #self.dropout10 = nn.Dropout(p=0.5, inplace=True)
        self.layer11 = torch.nn.Linear(128, 38)
        #self.dropout11 = nn.Dropout(p=0.5, inplace=True)
        self.layer12 = nn.Linear(38,1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        #x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        #x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        #x = self.dropout3(x)
        x = F.relu(self.layer4(x))
        #x = self.dropout4(x)
        x = F.relu(self.layer5(x))
        #x = self.dropout5(x)
        x = F.relu(self.layer6(x))
        #x = self.dropout6(x)
        x = F.relu(self.layer7(x))
        #x = self.dropout7(x)
        x = F.relu(self.layer8(x))
        #x = self.dropout8(x)
        x = F.relu(self.layer9(x))
        #x = self.dropout9(x)
        x = F.relu(self.layer10(x))
        #x = self.dropout10(x)
        x = F.relu(self.layer11(x))
        #x = self.dropout11(x)
        x = self.layer12(x)

        return x

class MLPReg3(nn.Module):
    def __init__(self, DO):
        super(MLPReg3, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(304, 152),
            nn.ReLU(inplace=True)
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg3Doal(nn.Module):
    def __init__(self, DO):
        super(MLPReg3Doal, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg3Doen(nn.Module):
    def __init__(self, DO):
        super(MLPReg3Doen, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg3Dode(nn.Module):
    def __init__(self, DO):
        super(MLPReg3Dode, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg4_ri(nn.Module):
    def __init__(self,DO):
        super(MLPReg4_ri, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.9)
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg4_rifea(nn.Module):
    def __init__(self,DO):
        super(MLPReg4_rifea, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.9)
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg4_rifea3(nn.Module):
    def __init__(self,DO):
        super(MLPReg4_rifea3, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.9)
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg4_rifea6(nn.Module):
    def __init__(self,DO):
        super(MLPReg4_rifea6, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(28, 56),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.9)
            nn.Linear(56, 112),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(112, 224),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(224, 224),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(224, 448),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(448, 448),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(448, 224),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(224, 224),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(224, 112),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(112, 56),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(56, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg4_rifea9(nn.Module):
    def __init__(self,DO):
        super(MLPReg4_rifea9, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.9)
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg4_rido(nn.Module):
    def __init__(self, DO):
        super(MLPReg4_rido, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg4_rinewdo(nn.Module):
    def __init__(self,DO):
        super(MLPReg4_rinewdo, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg4_ap(nn.Module):
    def __init__(self, DO):
        super(MLPReg4_ap, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg4_apfea(nn.Module):
    def __init__(self, DO):
        super(MLPReg4_apfea, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.9)
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg4_apdo(nn.Module):
    def __init__(self, DO):
        super(MLPReg4_apdo, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x



class MLPReg4_apnewdo(nn.Module):
    def __init__(self, DO):
        super(MLPReg4_apnewdo, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg4_riap(nn.Module):
    def __init__(self, DO):
        super(MLPReg4_riap, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 1216),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(1216, 1216),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(1216, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg4_riapfea(nn.Module):
    def __init__(self, DO):
        super(MLPReg4_riapfea, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg4_riapdo(nn.Module):
    def __init__(self, DO):
        super(MLPReg4_riapdo, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 1216),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(1216, 1216),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(1216, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg4_riapnewdo(nn.Module):
    def __init__(self, DO):
        super(MLPReg4_riapnewdo, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(76, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 1216),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(1216, 1216),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(1216, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1)
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg4Doal(nn.Module):
    def __init__(self, DO):
        super(MLPReg4Doal, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 72),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(72, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg4Doen(nn.Module):
    def __init__(self, DO):
        super(MLPReg4Doen, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(152, 72),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(72, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg4Dode(nn.Module):
    def __init__(self, DO):
        super(MLPReg4Dode, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 72),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(72, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg5(nn.Module):
    def __init__(self, DO):
        super(MLPReg5, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
        )
        self.regressor = nn.Linear(304, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5Doal(nn.Module):
    def __init__(self, DO):
        super(MLPReg5Doal, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
        )
        self.regressor = nn.Linear(304, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5Doen(nn.Module):
    def __init__(self, DO):
        super(MLPReg5Doen, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
        )
        self.regressor = nn.Linear(304, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5Dode(nn.Module):
    def __init__(self, DO):
        super(MLPReg5Dode, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
        )
        self.regressor = nn.Linear(304, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5b_ri(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_ri, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg5b_rinewdo(nn.Module):
    def __init__(self,DO):
        super(MLPReg5b_rinewdo, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(152, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x



class MLPReg5b_ap(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_ap, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5b_riap(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_riap, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(608, 304),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(304, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5b_rifea(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_rifea, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg5b_rifea3(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_rifea3, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg5b_rifea6(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_rifea6, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(28, 56),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(56, 112),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(112, 224),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(224, 224),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(224, 112),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(112, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg5b_rifea9(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_rifea9, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg5b_apfea(nn.Module):
    def __init__(self,DO):
        super(MLPReg5b_apfea, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg5b_riapfea(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_riapfea, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x




class MLPReg5bDoal(nn.Module):
    def __init__(self, DO):
        super(MLPReg5bDoal, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5bDoen(nn.Module):
    def __init__(self, DO):
        super(MLPReg5bDoen, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5bDode(nn.Module):
    def __init__(self, DO):
        super(MLPReg5bDode, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg5b_rido(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_rido, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5b_apdo(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_apdo, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg5b_riapdo(nn.Module):
    def __init__(self, DO):
        super(MLPReg5b_riapdo, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(304, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5c(nn.Module):
    def __init__(self, DO):
        super(MLPReg5c, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5cDoen(nn.Module):
    def __init__(self, DO):
        super(MLPReg5cDoen, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg5cDode(nn.Module):
    def __init__(self, DO):
        super(MLPReg5cDode, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg6(nn.Module):
    def __init__(self, DO):
        super(MLPReg6, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 1152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(1152, 1152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(1152, 1152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(1152, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 152),
            nn.ReLU(inplace=True)
        )
        self.regressor = nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg6c(nn.Module):
    def __init__(self, DO):
        super(MLPReg6c, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 1152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(1152, 1152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(1152, 1152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(1152, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            nn.Linear(76, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
        )
        self.regressor = nn.Linear(38, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg6cDoen(nn.Module):
    def __init__(self, DO):
        super(MLPReg6cDoen, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 1152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(1152, 1152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            nn.Linear(1152, 1152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            nn.Linear(1152, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            nn.Linear(76, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
        )
        self.regressor = nn.Linear(38, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg6cDode(nn.Module):
    def __init__(self, DO):
        super(MLPReg6cDode, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(608, 1152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=DO),
            nn.Linear(1152, 1152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            nn.Linear(1152, 1152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=DO),
            nn.Linear(1152, 608),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
            nn.Linear(76, 38),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DO),
        )
        self.regressor = nn.Linear(38, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x


class MLPReg7(nn.Module):
    def __init__(self, DO):
        super(MLPReg7, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 304),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Linear(304, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg8(nn.Module):
    def __init__(self, DO):
        super(MLPReg8, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Linear(608, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLPReg9(nn.Module):
    def __init__(self, DO):
        super(MLPReg9, self).__init__()
        self.rep_learner = nn.Sequential(
            nn.Linear(38, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            nn.Linear(608, 304),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Linear(304, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

class MLReg10a_ri(nn.Module):
    def __init__(self, DO):
        super(MLReg10a_ri, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 1216)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(1216, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.2),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.1),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True)

        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.regressor(x)

        return x


class MLReg10a_rido(nn.Module):
    def __init__(self, DO):
        super(MLReg10a_rido, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 1216)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(1216, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.2),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.1),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True)

        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.regressor(x)

        return x


class MLReg10a_ap(nn.Module):
    def __init__(self, DO):
        super(MLReg10a_ap, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 1216)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(1216, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.2),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.1),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True)

        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.regressor(x)

        return x


class MLReg10a_apdo(nn.Module):
    def __init__(self, DO):
        super(MLReg10a_apdo, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 1216)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(1216, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.2),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.1),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True)

        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.regressor(x)

        return x


class MLReg10a_riap(nn.Module):
    def __init__(self, DO):
        super(MLReg10a_riap, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 1216),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(1216, 2432)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(2432, 1216),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(1216, 608),
            nn.ReLU(inplace=True),
            torch.nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.2),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.1),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True)

        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.regressor(x)

        return x


class MLReg10a_riapdo(nn.Module):
    def __init__(self, DO):
        super(MLReg10a_riapdo, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            torch.nn.Linear(152, 304),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.9),
            torch.nn.Linear(304, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 1216),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(1216, 2432)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(2432, 1216),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(1216, 608),
            nn.ReLU(inplace=True),
            torch.nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.2),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.1),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True)

        )
        self.regressor = torch.nn.Linear(76, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.regressor(x)

        return x

#############################################################################################################
# Architecture Research
#############################################################################################################

# Reverse Triangle
class MLPArc1_ys(nn.Module):
    def __init__(self, DO):
        super(MLPArc1_ys, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(38, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(608, 608),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 152)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(152, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            torch.nn.Linear(76, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.2),
            torch.nn.Linear(76, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.1),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True)

        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.regressor(x)

        return x

class MLPArc1_ps(nn.Module):
    def __init__(self, DO):
        super(MLPArc1_ps, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 608),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(608, 304),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

# Rectangle
class MLPArc2_ys(nn.Module):
    def __init__(self, DO):
        super(MLPArc2_ys, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(38, 38)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.2),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.1),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True)

        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.regressor(x)

        return x

class MLPArc2_ps(nn.Module):
    def __init__(self, DO):
        super(MLPArc2_ps, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(38, 38),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(38, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x

# Autoencoder
class MLPArc3_ys(nn.Module):
    def __init__(self, DO):
        super(MLPArc3_ys, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Linear(38, 304),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(304, 152),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.9),
            torch.nn.Linear(152, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(152, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(152, 76)
        )
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(76, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True),
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.2),
            torch.nn.Linear(76, 76),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=.1),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True)

        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.regressor(x)

        return x

class MLPArc3_ps(nn.Module):
    def __init__(self, DO):
        super(MLPArc3_ps, self).__init__()
        self.rep_learner = nn.Sequential(
            torch.nn.Linear(38, 152),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(152, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(76, 38),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.2),
            torch.nn.Linear(38, 76),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=.1),
            torch.nn.Linear(76, 152),
            nn.ReLU(inplace=True),
        )
        self.regressor = torch.nn.Linear(152, 1)

    def forward(self, x):
        x = self.rep_learner(x)
        x = self.regressor(x)

        return x



#############################################################################################################
# Convolutional Neural Network with 1D Convolution layer
class CNN1DReg(nn.Module):
    def __init__(self):
        super(CNN1DReg, self).__init__()
        self.cnn1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=152, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.Dropout(p = 0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=152, out_channels=152, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.Dropout(p = 0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=152, out_channels=78, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.Dropout(p = 0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=78, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.Dropout(p = 0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=32, out_features=16, bias=True),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features=16, out_features=1, bias=True),
        )

    def forward(self, x):
        y = self.cnn1d(x)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

# Recurrent Neural Network
class RNNReg(nn.Module):
    def __init__(self):
        super(RNNReg, self).__init__()

        self.i2h = nn.Linear()

    def forward(self, x):

        return x
