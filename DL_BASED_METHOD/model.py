# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class IncBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, size = 15, stride = 1, padding = 7):
#         super(IncBlock,self).__init__()
        
#         self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias = False)
        
#         self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = size, stride = stride, padding = padding ),
#                                    nn.BatchNorm1d(out_channels//4))
        
#         self.conv2 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
#                                    nn.BatchNorm1d(out_channels//4),
#                                    nn.LeakyReLU(0.2),
#                                    nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size +2 , stride = stride, padding = padding + 1),
#                                    nn.BatchNorm1d(out_channels//4))
        
#         self.conv3 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
#                                    nn.BatchNorm1d(out_channels//4),
#                                    nn.LeakyReLU(0.2),
#                                    nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size + 4 , stride = stride, padding = padding + 2),
#                                    nn.BatchNorm1d(out_channels//4))
        
        
#         self.conv4 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
#                                    nn.BatchNorm1d(out_channels//4),
#                                    nn.LeakyReLU(0.2),
#                                    nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size + 6 , stride = stride, padding = padding + 3),
#                                    nn.BatchNorm1d(out_channels//4))
#         self.relu = nn.ReLU()
#     def forward(self,x):
#         res = self.conv1x1(x)
# #         print (res.size())

        
#         c1 = self.conv1(x)
# #         print (c1.size())
        
#         c2 = self.conv2(x)
# #         print (c2.size())
                
#         c3 = self.conv3(x)
# #         print (c3.size())
        
#         c4 = self.conv4(x)
# #         print (c4.size())
        
#         concat = torch.cat((c1,c2,c3,c4),dim = 1)
        
#         concat+=res
# #         print (concat.shape)
#         return self.relu(concat)

# class Unet(nn.Module):
#     def __init__(self, shape):
#         super(Unet, self).__init__()
#         in_channels = shape[1]

#         self.up1 = nn.Sequential(nn.ConvTranspose1d(in_channels,8,289, stride = 1, padding = 0),
#                                 nn.BatchNorm1d(8),
#                                 nn.LeakyReLU(0.2),
#                                 IncBlock(8,8))
#         self.up2 = nn.Sequential(nn.ConvTranspose1d(8,16,7, stride = 7, padding = 0),
#                                 nn.BatchNorm1d(16),
#                                 nn.LeakyReLU(0.2),
#                                 IncBlock(16,16))

        
        
        
        
#         self.en1 = nn.Sequential(nn.Conv1d(16, 32, 3, padding = 1), 
#                                 nn.BatchNorm1d(32),
#                                 nn.LeakyReLU(0.2),
#                                 nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
#                                 IncBlock(32,32))
        
#         self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.LeakyReLU(0.2),
#                                  nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
#                                 IncBlock(64,64))
        
              
#         self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
#                                  nn.BatchNorm1d(128),
#                                  nn.LeakyReLU(0.2),
#                                  nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
#                                 IncBlock(128,128))
        
#         self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
#                                  nn.BatchNorm1d(256),
#                                  nn.LeakyReLU(0.2),
#                                  nn.Conv1d(256, 256, 5, stride = 2, padding = 1),
#                                 IncBlock(256,256))
        
        
#         self.en5 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
#                                  nn.BatchNorm1d(512),
#                                  nn.LeakyReLU(0.2),
#                                  IncBlock(512,512))
        
        
#         self.de1 = nn.Sequential(nn.ConvTranspose1d(512,256,1),
#                                nn.BatchNorm1d(256),
#                                nn.LeakyReLU(0.2),
#                                 IncBlock(256,256))
        
#         self.de2 =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
#                                nn.BatchNorm1d(256),
#                                nn.LeakyReLU(0.2),
#                                   nn.ConvTranspose1d(256,128,3, stride = 2),
#                                 IncBlock(128,128))
        
#         self.de3 =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
#                                nn.BatchNorm1d(128),
#                                nn.LeakyReLU(0.2),
#                                 nn.ConvTranspose1d(128,64,3, stride = 2),
#                                 IncBlock(64,64))
        
#         self.de4 =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
#                                nn.BatchNorm1d(64),
#                                nn.LeakyReLU(0.2),
#                                 nn.ConvTranspose1d(64,32,3, stride = 2),
#                                 IncBlock(32,32))
        
#         self.de5 = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding = 1),
#                                nn.BatchNorm1d(32),
#                                nn.LeakyReLU(0.2),
#                                 nn.ConvTranspose1d(32,16,3, stride = 2),
#                                 IncBlock(16,16))
                               
#         self.de6 = nn.Sequential(nn.ConvTranspose1d(16,1,1,stride =1),
#                                 nn.LeakyReLU(0.2))

#     def forward(self,x):
        
# #         x = self.inter(x)
# #         x = nn.ConstantPad1d((1,1),0)(x)
# #         print ("inp: ",x.shape)
#         up1 = self.up1(x)
# #         print ("up1: ", up1.shape)
        
#         up2 = self.up2(up1)
# #         print ("up2: ", up2.shape)
        
#         e1 = self.en1(up2)
# #         print ("e1: ", e1.shape)
        
#         e2 = self.en2(e1)
# #         print ("e2: ", e2.shape)
        
#         e3 = self.en3(e2)
# #         print ("e3: ",e3.shape)
        
#         e4 = self.en4(e3)
# #         print ("e4: ", e4.shape)
        
#         e5 = self.en5(e4)
# #         print ("e5 :", e5.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
#         d1 = self.de1(e5)
# #         print ("d1: ",d1.shape)
        
        
#         cat = torch.cat([d1,e4],1)
#         d2 = self.de2(cat)
#         d2 = F.pad(d2,(0,1))
# #         print ("d2: ", d2.shape)
        
#         cat = torch.cat([d2,e3],1)
        
#         d3 = self.de3(cat)
#         d3 = d3 [:,:,:-1]
# #         print ("d3: ",d3.shape)
        
#         cat = torch.cat([d3,e2],1)

#         d4 = self.de4(cat)
#         d4 = d4[:,:,:-1]
# #         print ("d4: " ,d4.shape)
        
#         cat = torch.cat([d4,e1],1)
        
#         d5 = self.de5(cat)
#         d5 = d5[:,:,:-1]
        
        
        
# #         print ("d5: ",d5.shape)

#         d6 = self.de6(d5)
# #         print ("d6: ", d6.shape)

#         return d6

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class IncBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, size = 15, stride = 1, padding = 7):
#         super(IncBlock,self).__init__()
        
#         self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias = False)
        
#         self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = size, stride = stride, padding = padding ),
#                                    nn.BatchNorm1d(out_channels//4))
        
#         self.conv2 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
#                                    nn.BatchNorm1d(out_channels//4),
#                                    nn.LeakyReLU(0.2),
#                                    nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size +2 , stride = stride, padding = padding + 1),
#                                    nn.BatchNorm1d(out_channels//4))
        
#         self.conv3 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
#                                    nn.BatchNorm1d(out_channels//4),
#                                    nn.LeakyReLU(0.2),
#                                    nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size + 4 , stride = stride, padding = padding + 2),
#                                    nn.BatchNorm1d(out_channels//4))
        
        
#         self.conv4 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
#                                    nn.BatchNorm1d(out_channels//4),
#                                    nn.LeakyReLU(0.2),
#                                    nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size + 6 , stride = stride, padding = padding + 3),
#                                    nn.BatchNorm1d(out_channels//4))
#         self.relu = nn.ReLU()
#     def forward(self,x):
#         res = self.conv1x1(x)
# #         print (res.size())

        
#         c1 = self.conv1(x)
# #         print (c1.size())
        
#         c2 = self.conv2(x)
# #         print (c2.size())
                
#         c3 = self.conv3(x)
# #         print (c3.size())
        
#         c4 = self.conv4(x)
# #         print (c4.size())
        
#         concat = torch.cat((c1,c2,c3,c4),dim = 1)
        
#         concat+=res
# #         print (concat.shape)
#         return self.relu(concat)

# class Unet(nn.Module):
#     def __init__(self, shape):
#         super(Unet, self).__init__()
#         in_channels = shape[1]

#         self.up1 = nn.Sequential(nn.ConvTranspose1d(in_channels,8,289, stride = 1, padding = 0),
#                                 nn.BatchNorm1d(8),
#                                 nn.LeakyReLU(0.2),
#                                 IncBlock(8,8))
#         self.up2 = nn.Sequential(nn.ConvTranspose1d(8,16,7, stride = 7, padding = 0),
#                                 nn.BatchNorm1d(16),
#                                 nn.LeakyReLU(0.2),
#                                 IncBlock(16,16))

        
#         self.ea_conv1 = nn.Sequential(nn.Conv2d(1, 1, (3,3), padding = (0,1)),
#                                      nn.BatchNorm2d(1),
#                                      nn.LeakyReLU(0.2))
#         self.mp1 = nn.MaxPool2d((2,1))
        
#         self.ea_conv2 = nn.Sequential(nn.Conv2d(1, 1, (5,5), padding = (0,2)),
#                                      nn.BatchNorm2d(1),
#                                      nn.LeakyReLU(0.2))
        
#         self.ea_conv3 = nn.Sequential(nn.Conv2d(1, 1, (3,3), padding = (0,1)),
#                                      nn.BatchNorm2d(1),
#                                      nn.LeakyReLU(0.2))
        
#         self.en1 = nn.Sequential(nn.Conv1d(1, 32, 3, padding = 1), 
#                                 nn.BatchNorm1d(32),
#                                 nn.LeakyReLU(0.2),
#                                 nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
#                                 IncBlock(32,32))
        
#         self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
#                                 nn.BatchNorm1d(64),
#                                 nn.LeakyReLU(0.2),
#                                  nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
#                                 IncBlock(64,64))
        
              
#         self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
#                                  nn.BatchNorm1d(128),
#                                  nn.LeakyReLU(0.2),
#                                  nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
#                                 IncBlock(128,128))
        
#         self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
#                                  nn.BatchNorm1d(256),
#                                  nn.LeakyReLU(0.2),
#                                  nn.Conv1d(256, 256, 5, stride = 2, padding = 1),
#                                 IncBlock(256,256))
        
        
#         self.en5 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
#                                  nn.BatchNorm1d(512),
#                                  nn.LeakyReLU(0.2),
#                                  IncBlock(512,512))
#         self.en6 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1, stride  = 2),
#                                  nn.BatchNorm1d(1024),
#                                  nn.LeakyReLU(0.2),
#                                  IncBlock(1024,1024))
        
#         self.de1 = nn.Sequential(nn.ConvTranspose1d(1024,512,1, stride = 2),
#                                nn.BatchNorm1d(512),
#                                nn.LeakyReLU(0.2),
#                                 IncBlock(512,512))
        
#         self.de2 = nn.Sequential(nn.ConvTranspose1d(1024,512,1),
#                                nn.BatchNorm1d(512),
#                                nn.LeakyReLU(0.2),
#                             nn.ConvTranspose1d(512,256,3, stride = 1, padding = 1),
#                                 IncBlock(256,256))
        
#         self.de3 =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
#                                nn.BatchNorm1d(256),
#                                nn.LeakyReLU(0.2),
#                                   nn.ConvTranspose1d(256,128,3,stride = 2),
#                                 IncBlock(128,128))
        
#         self.de4 =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
#                                nn.BatchNorm1d(128),
#                                nn.LeakyReLU(0.2),
#                                 nn.ConvTranspose1d(128,64,3, stride = 2),
#                                 IncBlock(64,64))
        
#         self.de5 =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
#                                nn.BatchNorm1d(64),
#                                nn.LeakyReLU(0.2),
#                                 nn.ConvTranspose1d(64,32,3, stride = 2),
#                                 IncBlock(32,32))
        
#         self.de6 = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding =1),
#                                nn.BatchNorm1d(32),
#                                nn.LeakyReLU(0.2),
#                                 nn.ConvTranspose1d(32,16,3, stride = 2, padding = 0),
#                                 IncBlock(16,16))
                               
#         self.de7 = nn.Sequential(nn.ConvTranspose1d(16,1,1,stride =1, padding = 0),
#                                 nn.LeakyReLU(0.2))

#     def forward(self,x):
        
# #         x = self.inter(x)
# #         x = nn.ConstantPad1d((1,1),0)(x)
# #         print ("inp: ",x.shape)
#         up1 = self.up1(x)
# #         print ("up1: ", up1.shape)
        
#         up2 = self.up2(up1)
# #         print ("up2: ", up2.shape)
#         make_2d = up2.unsqueeze(1)
# #         print ("up2 after unsqueezing: ", make_2d.shape)
#         avg1 = self.mp1(self.ea_conv1(make_2d))
#         avg2 = self.ea_conv2(avg1)
#         avg3 = self.ea_conv3(avg2)
# #         print(avg1.shape)
# #         print (avg2.shape)
# #         print (avg3.shape)
#         avg3 = avg3.squeeze(1)

#         e1 = self.en1(avg3)
# #         print ("e1: ", e1.shape)
        
#         e2 = self.en2(e1)
# #         print ("e2: ", e2.shape)
# #         
#         e3 = self.en3(e2)
# #         print ("e3: ",e3.shape)
        
#         e4 = self.en4(e3)
# #         print ("e4: ", e4.shape)
        
#         e5 = self.en5(e4)
# #         print ("e5 :", e5.shape)
#         e6 = self.en6(e5)
# #         print ("e6 :", e6.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
#         d1 = self.de1(e6)
# #         print ("d1: ",d1.shape)
        
        
#         cat = torch.cat([d1,e5],1)
#         d2 = self.de2(cat)
#         d2 = F.pad(d2,(0,1))
# #         print ("d2: ", d2.shape)
        
#         cat = torch.cat([d2[:,:,:-1],e4],1)
        
#         d3 = self.de3(cat)
#         d3 = F.pad(d3, (0,1))
# #         print ("d3: ",d3.shape)
        
#         cat = torch.cat([d3,e3],1)

#         d4 = self.de4(cat)
#         d4 = d4[:,:,:-1]
# #         print ("d4: " ,d4.shape)
        
#         cat = torch.cat([d4,e2],1)
        
#         d5 = self.de5(cat)
#         d5 = d5[:,:,:-1]
        
        
        
# #         print ("d5: ",d5.shape)
#         cat = torch.cat([d5,e1],1)
#         d6 = self.de6(cat)[:,:,:-1]
# #         print ("d6: ", d6.shape)
#         d7 = self.de7(d6)
# #         print ("d7: ", d7.shape)
#         return d7
import torch
import torch.nn as nn
import torch.nn.functional as F

class IncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size = 15, stride = 1, padding = 7):
        super(IncBlock,self).__init__()
        
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias = False)
        
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = size, stride = stride, padding = padding ),
                                   nn.BatchNorm1d(out_channels//4))
        
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
                                   nn.BatchNorm1d(out_channels//4),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size +2 , stride = stride, padding = padding + 1),
                                   nn.BatchNorm1d(out_channels//4))
        
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
                                   nn.BatchNorm1d(out_channels//4),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size + 4 , stride = stride, padding = padding + 2),
                                   nn.BatchNorm1d(out_channels//4))
        
        
        self.conv4 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, kernel_size = 1, bias = False),
                                   nn.BatchNorm1d(out_channels//4),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv1d(out_channels//4, out_channels//4, kernel_size = size + 6 , stride = stride, padding = padding + 3),
                                   nn.BatchNorm1d(out_channels//4))
        self.relu = nn.ReLU()
    def forward(self,x):
        res = self.conv1x1(x)
#         print (res.size())

        
        c1 = self.conv1(x)
#         print (c1.size())
        
        c2 = self.conv2(x)
#         print (c2.size())
                
        c3 = self.conv3(x)
#         print (c3.size())
        
        c4 = self.conv4(x)
#         print (c4.size())
        
        concat = torch.cat((c1,c2,c3,c4),dim = 1)
        
        concat+=res
#         print (concat.shape)
        return self.relu(concat)

class Unet(nn.Module):
    def __init__(self, shape):
        super(Unet, self).__init__()
        in_channels = shape[1]

        self.up1 = nn.Sequential(nn.ConvTranspose1d(in_channels,4,3, stride = 2,padding = 0),
                                nn.BatchNorm1d(4),
                                nn.LeakyReLU(0.2),
                                IncBlock(4,4))
        self.up2 = nn.Sequential(nn.ConvTranspose1d(4,8,3,stride = 2, padding = 0),
                                nn.BatchNorm1d(8),
                                nn.LeakyReLU(0.2),
                                IncBlock(8,8))
        self.up3 = nn.Sequential(nn.ConvTranspose1d(8,16,3,stride = 2, padding = 0),
                                nn.BatchNorm1d(16),
                                nn.LeakyReLU(0.2),
                                IncBlock(16,16))
        
        
        self.en1 = nn.Sequential(nn.Conv1d(16, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 5, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        
        self.en5 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 5, stride = 2, padding = 1),
                                 IncBlock(512,512))
        self.en6 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1, stride  = 2),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(1024,1024))
        
        self.de1 = nn.Sequential(nn.ConvTranspose1d(1024,512,1, stride = 2),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(0.2),
                                IncBlock(512,512))
        
        self.de2 = nn.Sequential(nn.ConvTranspose1d(1024,512,1),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(0.2),
                            nn.ConvTranspose1d(512,256,3, stride = 2, padding = 0),
                                IncBlock(256,256))
        
        self.de3 =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
                               nn.BatchNorm1d(256),
                               nn.LeakyReLU(0.2),
                                  nn.ConvTranspose1d(256,128,3,stride = 2),
                                IncBlock(128,128))
        
        self.de4 =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(128),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(128,64,3, stride = 2),
                                IncBlock(64,64))
        
        self.de5 =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(64),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(64,32,3, stride = 2),
                                IncBlock(32,32))
        
        self.de6 = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding =1),
                               nn.BatchNorm1d(32),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(32,16,3, stride = 2, padding = 0),
                                IncBlock(16,16))
                               
        self.de7 = nn.Sequential(nn.ConvTranspose1d(16,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))

    def forward(self,x):

#         print ("inp: ",x.shape)
        up1 = self.up1(x)[:,:,:-1]
#         print ("up1: ", up1.shape)
        
        up2 = self.up2(up1)[:,:,:-1]
#         print ("up2: ", up2.shape)
        
        up3 = self.up3(up2)[:,:,:-1]
#         print ("up3: ", up3.shape)

        e1 = self.en1(up3)
#         print ("e1: ", e1.shape)
        
        e2 = self.en2(e1)
#         print ("e2: ", e2.shape)
# #         
        e3 = self.en3(e2)
#         print ("e3: ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("e4: ", e4.shape)
        
        e5 = self.en5(e4)
#         print ("e5 :", e5.shape)

        e6 = self.en6(e5)
#         print ("e6 :", e6.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
        d1 = self.de1(e6)
#         print ("d1: ",d1.shape)  

        cat = torch.cat([d1,e5],1)
    
        d2 = self.de2(cat)
#         print ("d2: ", d2.shape)
        
        cat = torch.cat([d2,e4],1)
        
        d3 = self.de3(cat)
        d3 = F.pad(d3, (0,1))
#         print ("d3: ",d3.shape)
        
        cat = torch.cat([d3,e3],1)

        d4 = self.de4(cat)
        d4 = d4[:,:,:-1]
#         print ("d4: " ,d4.shape)

        cat = torch.cat([d4,e2],1)
        
        d5 = self.de5(cat)
        d5 = d5[:,:,:-1]
#         print ("d5: ",d5.shape)
        
        cat = torch.cat([d5,e1],1)
        
        d6 = self.de6(cat)[:,:,:-1]
#         print ("d6: ", d6.shape)
        
        d7 = self.de7(d6)
#         print ("d7: ", d7.shape)
        
        return d7
    
class UnetEncoder(nn.Module):
    def __init__(self, shape):
        super(UnetEncoder, self).__init__()
        
        in_channels = shape[1]

        self.up1 = nn.Sequential(nn.ConvTranspose1d(in_channels,4,3, stride = 2,padding = 0),
                                nn.BatchNorm1d(4),
                                nn.LeakyReLU(0.2),
                                IncBlock(4,4))
        self.up2 = nn.Sequential(nn.ConvTranspose1d(4,8,3,stride = 2, padding = 0),
                                nn.BatchNorm1d(8),
                                nn.LeakyReLU(0.2),
                                IncBlock(8,8))
        self.up3 = nn.Sequential(nn.ConvTranspose1d(8,16,3,stride = 2, padding = 0),
                                nn.BatchNorm1d(16),
                                nn.LeakyReLU(0.2),
                                IncBlock(16,16))
        
        
        self.en1 = nn.Sequential(nn.Conv1d(16, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 5, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        
        self.en5 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 5, stride = 2, padding = 1),
                                 IncBlock(512,512))
        self.en6 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1, stride  = 2),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(1024,1024))
        
        self.f1 = nn.Sequential(nn.Linear(1024*64, 1024,bias = True),
                               nn.BatchNorm1d(1024),
                                nn.LeakyReLU(0.2),
                                nn.Dropout(0.5)
                                 )
        self.f2 = nn.Sequential(nn.Linear(1024, 512,bias = True),
                               nn.BatchNorm1d(512),
                                nn.LeakyReLU(0.2),
                                nn.Dropout(0.5)
                                 )
        self.f3 = nn.Sequential(nn.Linear(512, 1,bias = True))
        
        

        

    def forward(self,x):

        x = torch.unsqueeze(x[:,0], dim= 1)
#         print ("inp: ",x.shape)

        up1 = self.up1(x)[:,:,:-1]
#         print ("up1: ", up1.shape)
        
        up2 = self.up2(up1)[:,:,:-1]
#         print ("up2: ", up2.shape)
        
        up3 = self.up3(up2)[:,:,:-1]
#         print ("up3: ", up3.shape)

        e1 = self.en1(up3)
#         print ("e1: ", e1.shape)
        
        e2 = self.en2(e1)
#         print ("e2: ", e2.shape)
# #         
        e3 = self.en3(e2)
#         print ("e3: ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("e4: ", e4.shape)
        
        e5 = self.en5(e4)
#         print ("e5 :", e5.shape)

        e6 = self.en6(e5)
#         print ("e6 :", e6.shape)
#         print ("-----------------------------------------------------------------------------")
#         -----------------------------------------------------------------------------
        flatten = e6.view(e6.shape[0],e6.shape[1]*e6.shape[2])
#         print(flatten.shape)
        out = self.f3(self.f2(self.f1(flatten)))
        return out
    
class HRUnet(nn.Module):
    def __init__(self, shape):
        super(HRUnet, self).__init__()
        in_channels = shape[1]
        
        
        
        self.en1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 5, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        
        self.en5 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 5, stride = 2, padding = 1),
                                 IncBlock(512,512))
        self.en6 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1, stride  = 2),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(1024,1024))
        
        self.de1_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1, stride = 2),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(0.2),
                                IncBlock(512,512))
        
        self.de2_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(0.2),
                            nn.ConvTranspose1d(512,256,3, stride = 2, padding = 0),
                                IncBlock(256,256))
        
        self.de3_ecg =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
                               nn.BatchNorm1d(256),
                               nn.LeakyReLU(0.2),
                                  nn.ConvTranspose1d(256,128,3,stride = 2),
                                IncBlock(128,128))
        
        self.de4_ecg =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(128),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(128,64,3, stride = 2),
                                IncBlock(64,64))
        
        self.de5_ecg =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(64),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(64,32,3, stride = 2),
                                IncBlock(32,32))
        
        self.de6_ecg = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding =1),
                               nn.BatchNorm1d(32),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(32,16,3, stride = 2, padding = 0),
                                IncBlock(16,16))
                               
        self.de7_ecg = nn.Sequential(nn.ConvTranspose1d(16,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        self.de8_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        self.de9_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        
        

    def forward(self,x):
#         print (x.shape)
        e1 = self.en1(x)
#         print ("e1: ", e1.shape)
        
        e2 = self.en2(e1)
#         print ("e2: ", e2.shape)
# #         
        e3 = self.en3(e2)
#         print ("e3: ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("e4: ", e4.shape)
        
        e5 = self.en5(e4)
#         print ("e5 :", e5.shape)

        e6 = self.en6(e5)
#         print ("e6 :", e6.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
        d1_ecg = self.de1_ecg(e6)
#         print ("d1: ",d1.shape)  


        cat_ecg = torch.cat([d1_ecg,e5],1)
    

        d2_ecg = self.de2_ecg(cat_ecg)
#         print ("d2: ", d2.shape)
        

        cat_ecg = torch.cat([d2_ecg,e4],1)
        
        
        d3_ecg = self.de3_ecg(cat_ecg)
        d3_ecg = F.pad(d3_ecg, (0,1))
#         print ("d3: ",d3.shape)
        
        cat_ecg = torch.cat([d3_ecg,e3],1)

        
        d4_ecg = self.de4_ecg(cat_ecg)
        d4_ecg = d4_ecg[:,:,:-1]
#         print ("d4: " ,d4.shape)

        cat_ecg = torch.cat([d4_ecg,e2],1)
        
        
        d5_ecg = self.de5_ecg(cat_ecg)
        d5_ecg = d5_ecg[:,:,:-1]
#         print ("d5: ",d5.shape)
        
        cat_ecg = torch.cat([d5_ecg,e1],1)
        
        d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
#         print ("d6: ", d6.shape)
        
        d7_ecg = self.de7_ecg(d6_ecg)
#         print ("d7: ", d7.shape)
        d8_ecg = self.de8_ecg(d7_ecg)
        
        d9_ecg = self.de9_ecg(d8_ecg)
        
        return d9_ecg
    
class BRUnet(nn.Module):
    def __init__(self, shape):
        super(BRUnet, self).__init__()
        in_channels = shape[1]
        
        
        
        self.en1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 5, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        
        self.en5 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 5, stride = 2, padding = 1),
                                 IncBlock(512,512))
        self.en6 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1, stride  = 2),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(1024,1024))
        
        self.de1_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1, stride = 2),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(0.2),
                                IncBlock(512,512))
        
        self.de2_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(0.2),
                            nn.ConvTranspose1d(512,256,3, stride = 2, padding = 0),
                                IncBlock(256,256))
        
        self.de3_ecg =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
                               nn.BatchNorm1d(256),
                               nn.LeakyReLU(0.2),
                                  nn.ConvTranspose1d(256,128,3,stride = 2),
                                IncBlock(128,128))
        
        self.de4_ecg =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(128),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(128,64,3, stride = 2),
                                IncBlock(64,64))
        
        self.de5_ecg =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(64),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(64,32,3, stride = 2),
                                IncBlock(32,32))
        
        self.de6_ecg = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding =1),
                               nn.BatchNorm1d(32),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(32,16,3, stride = 2, padding = 0),
                                IncBlock(16,16))
                               
        self.de7_ecg = nn.Sequential(nn.ConvTranspose1d(16,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        self.de8_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        self.de9_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        
        

    def forward(self,x):
#         print (x.shape)
        e1 = self.en1(x)
#         print ("e1: ", e1.shape)
        
        e2 = self.en2(e1)
#         print ("e2: ", e2.shape)
# #         
        e3 = self.en3(e2)
#         print ("e3: ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("e4: ", e4.shape)
        
        e5 = self.en5(e4)
#         print ("e5 :", e5.shape)
        
        e6 = self.en6(e5)
        
#         print ("e6 :", e6.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
        d1_ecg = self.de1_ecg(e6)
#         print ("d1: ",d1.shape)  

        cat_ecg = torch.cat([d1_ecg,e5],1)
        

        d2_ecg = self.de2_ecg(cat_ecg)
#         print ("d2: ", d2.shape)
        

        cat_ecg = torch.cat([d2_ecg,e4],1)
        
        
        d3_ecg = self.de3_ecg(cat_ecg)
        d3_ecg = F.pad(d3_ecg, (0,1))
#         print ("d3: ",d3.shape)
        
        cat_ecg = torch.cat([d3_ecg,e3],1)

        
        d4_ecg = self.de4_ecg(cat_ecg)
        d4_ecg = d4_ecg[:,:,:-1]
#         print ("d4: " ,d4.shape)

        cat_ecg = torch.cat([d4_ecg,e2],1)
        
        
        d5_ecg = self.de5_ecg(cat_ecg)
        d5_ecg = d5_ecg[:,:,:-1]
#         print ("d5: ",d5.shape)
        
        cat_ecg = torch.cat([d5_ecg,e1],1)
        
        d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
#         print ("d6: ", d6.shape)
        
        d7_ecg = self.de7_ecg(d6_ecg)
#         print ("d7: ", d7.shape)
        d8_ecg = self.de8_ecg(d7_ecg)
        
        d9_ecg = self.de9_ecg(d8_ecg)
        
        return d9_ecg

    
class BRUnet_mod(nn.Module):
    def __init__(self, shape):
        super(BRUnet_mod, self).__init__()
        in_channels = shape[1]
        
        
        
        self.en1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 4, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        
        self.en5 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 3,padding = 1),
                                 IncBlock(512,512))
        
        self.en6 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(1024,1024))
        
        self.de1_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1, stride = 1),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(0.2),
                                IncBlock(512,512))
        
        self.de2_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(0.2),
                            nn.ConvTranspose1d(512,256,1),
                                IncBlock(256,256))
        
        self.de3_ecg =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
                               nn.BatchNorm1d(256),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(256,128,4,stride = 2, padding =1),
                                IncBlock(128,128))
        
        self.de4_ecg =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(128),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(128,64,3, stride = 2),
                                IncBlock(64,64))
        
        self.de5_ecg =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(64),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(64,32,3, stride = 2),
                                IncBlock(32,32))
        
        self.de6_ecg = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding =1),
                               nn.BatchNorm1d(32),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(32,16,3, stride = 2, padding = 0),
                                IncBlock(16,16))
                               
        self.de7_ecg = nn.Sequential(nn.ConvTranspose1d(16,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        self.de8_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        self.de9_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        
        

    def forward(self,x):
#         print (x.shape)
        e1 = self.en1(x)
#         print ("e1: ", e1.shape)
        
        e2 = self.en2(e1)
#         print ("e2: ", e2.shape)
# #         
        e3 = self.en3(e2)
#         print ("e3: ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("e4: ", e4.shape)
        
        e5 = self.en5(e4)
#         print ("e5 :", e5.shape)
        
        e6 = self.en6(e5)
#         print ("e6 :", e6.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
        d1_ecg = self.de1_ecg(e6)
#         print ("d1: ",d1.shape)  


        cat_ecg = torch.cat([d1_ecg,e5],1)
    

        d2_ecg = self.de2_ecg(cat_ecg)
#         print ("d2: ", d2.shape)
        

        cat_ecg = torch.cat([d2_ecg,e4],1)
        
        
        d3_ecg = self.de3_ecg(cat_ecg)
        #d3_ecg = F.pad(d3_ecg, (0,1))
#         print ("d3: ",d3.shape)
        
        cat_ecg = torch.cat([d3_ecg,e3],1)

        
        d4_ecg = self.de4_ecg(cat_ecg)
        d4_ecg = d4_ecg[:,:,:-1]
#         print ("d4: " ,d4.shape)

        cat_ecg = torch.cat([d4_ecg,e2],1)
        
        
        d5_ecg = self.de5_ecg(cat_ecg)
        d5_ecg = d5_ecg[:,:,:-1]
#         print ("d5: ",d5.shape)
        
        cat_ecg = torch.cat([d5_ecg,e1],1)
        
        d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
#         print ("d6: ", d6.shape)
        
        d7_ecg = self.de7_ecg(d6_ecg)
#         print ("d7: ", d7.shape)
        d8_ecg = self.de8_ecg(d7_ecg)
       
        d9_ecg = self.de9_ecg(d8_ecg)
        
        return d9_ecg

class BRUnet_raw_mod(nn.Module):
    def __init__(self, shape):
        super(BRUnet_raw_mod, self).__init__()
        in_channels = shape[1]
        
        
        
        self.en1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 4, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        self.en5 = nn.Sequential(nn.Conv1d(256,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 4, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        self.en6 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 4, stride = 2, padding = 1),
                                IncBlock(512,512))
        
        self.en7 = nn.Sequential(nn.Conv1d(512,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 4, stride = 2, padding = 1),
                                IncBlock(512,512))
        
        self.en8 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(1024, 1024, 4, stride = 2, padding = 1),
                                IncBlock(1024,1024))
       
        self.de1_ecg =  nn.Sequential(nn.Conv1d(1024,512,3, padding = 1),
                                nn.BatchNorm1d(512),
                                nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(512,512,4,stride = 2, padding =1),
                                IncBlock(512,512))

        self.de2_ecg =  nn.Sequential(nn.Conv1d(1024,512,3, padding = 1),
                                nn.BatchNorm1d(512),
                                nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(512,512,4,stride = 2, padding =1),
                                IncBlock(512,512))

        self.de3_ecg =  nn.Sequential(nn.Conv1d(1024,256,3, padding = 1),
                                nn.BatchNorm1d(256),
                                nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(256,256,4,stride = 2, padding =1),
                                IncBlock(256,256))

        self.de4_ecg =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(0.2),
                                        nn.ConvTranspose1d(256,256,4,stride = 2, padding =1),
                                        IncBlock(256,256))

        self.de5_ecg = nn.Sequential(nn.ConvTranspose1d(512,256,1),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(0.2),
                        nn.ConvTranspose1d(256,256,1),
                        IncBlock(256,256))
        
        self.de6_ecg = nn.Sequential(nn.ConvTranspose1d(256,128,1),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2),
                        nn.ConvTranspose1d(128,64,1),
                        IncBlock(64,64))
        
        self.de7_ecg = nn.Sequential(nn.ConvTranspose1d(64,8,1),
                        nn.BatchNorm1d(8),
                        nn.LeakyReLU(0.2),
                        nn.ConvTranspose1d(8,4,1),
                        IncBlock(4,4))

        self.de8_ecg = nn.Sequential(nn.ConvTranspose1d(4,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        self.de9_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        
        

    def forward(self,x):
#         print (x.shape)
        e1 = self.en1(x)
#         print ("e1: ", e1.shape)
        
        e2 = self.en2(e1)
#         print ("e2: ", e2.shape)
# #         
        e3 = self.en3(e2)
#         print ("e3: ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("e4: ", e4.shape)
        
        e5 = self.en5(e4)
#         print ("e5 :", e5.shape)
        
        e6 = self.en6(e5)

        e7 = self.en7(e6)
        
        e8 = self.en8(e7)

#         print ("e6 :", e6.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
        d1_ecg = self.de1_ecg(e8)
#         print ("d1: ",d1.shape)  

        cat_ecg = torch.cat([d1_ecg,e7],1)
    
        d2_ecg = self.de2_ecg(cat_ecg)
#         print ("d2: ", d2.shape)
        
        cat_ecg = torch.cat([d2_ecg,e6],1)
        
        
        d3_ecg = self.de3_ecg(cat_ecg)
        #d3_ecg = F.pad(d3_ecg, (0,1))
#         print ("d3: ",d3.shape)
        
        cat_ecg = torch.cat([d3_ecg,e5],1)

        
        d4_ecg = self.de4_ecg(cat_ecg)

        cat_ecg = torch.cat([d4_ecg,e4],1)
        
        
        d5_ecg = self.de5_ecg(cat_ecg)
        d6_ecg = self.de6_ecg(d5_ecg)
        d7_ecg = self.de7_ecg(d6_ecg)
        d8_ecg = self.de8_ecg(d7_ecg)
        d9_ecg = self.de9_ecg(d8_ecg)
        
        return d9_ecg


class BRUnet_Multi(nn.Module):
    def __init__(self, shape):
        super(BRUnet_Multi, self).__init__()
        in_channels = shape[1]
        
        
        
        self.en1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 4, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        
        self.en5 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 3,padding = 1),
                                 IncBlock(512,512))
        
        self.en6 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(1024,1024))

        self.en7_p = nn.Sequential(nn.Conv1d(1024, 128, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(128,128))
        self.en8_p = nn.Sequential(nn.Conv1d(128, 64, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(64),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(64,64))

        self.en9_p = nn.Sequential(nn.Conv1d(64, 4, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(4),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(4,4))
        self.fc = nn.Linear(4,1)      
        
        self.de1_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1, stride = 1),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(0.2),
                                IncBlock(512,512))
        
        self.de2_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1),
                               nn.BatchNorm1d(512),
                               nn.LeakyReLU(0.2),
                            nn.ConvTranspose1d(512,256,1),
                                IncBlock(256,256))
        
        self.de3_ecg =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
                               nn.BatchNorm1d(256),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(256,128,4,stride = 2, padding =1),
                                IncBlock(128,128))
        
        self.de4_ecg =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(128),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(128,64,3, stride = 2),
                                IncBlock(64,64))
        
        self.de5_ecg =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
                               nn.BatchNorm1d(64),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(64,32,3, stride = 2),
                                IncBlock(32,32))
        
        self.de6_ecg = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding =1),
                               nn.BatchNorm1d(32),
                               nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(32,16,3, stride = 2, padding = 0),
                                IncBlock(16,16))
                               
        self.de7_ecg = nn.Sequential(nn.ConvTranspose1d(16,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        self.de8_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        self.de9_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        
        

    def forward(self,x):
#         print (x.shape)
        e1 = self.en1(x)
#         print ("e1: ", e1.shape)
        
        e2 = self.en2(e1)
#         print ("e2: ", e2.shape)
# #         
        e3 = self.en3(e2)
#         print ("e3: ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("e4: ", e4.shape)
        
        e5 = self.en5(e4)
#         print ("e5 :", e5.shape)
        e6 = self.en6(e5)

        out_1 = self.en7_p(e6)

        out_2 =  self.en8_p(out_1)
        #import pdb;pdb.set_trace()
        out_3 =  self.en9_p(out_2)
        out_4 = self.fc(torch.squeeze(out_3,-1))
#         print ("e6 :", e6.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
        d1_ecg = self.de1_ecg(e6)
#         print ("d1: ",d1.shape)  


        cat_ecg = torch.cat([d1_ecg,e5],1)
    

        d2_ecg = self.de2_ecg(cat_ecg)
#         print ("d2: ", d2.shape)
        

        cat_ecg = torch.cat([d2_ecg,e4],1)
        
        
        d3_ecg = self.de3_ecg(cat_ecg)
        #d3_ecg = F.pad(d3_ecg, (0,1))
#         print ("d3: ",d3.shape)
        
        cat_ecg = torch.cat([d3_ecg,e3],1)

        
        d4_ecg = self.de4_ecg(cat_ecg)
        d4_ecg = d4_ecg[:,:,:-1]
#         print ("d4: " ,d4.shape)

        cat_ecg = torch.cat([d4_ecg,e2],1)
        
        
        d5_ecg = self.de5_ecg(cat_ecg)
        d5_ecg = d5_ecg[:,:,:-1]
#         print ("d5: ",d5.shape)
        
        cat_ecg = torch.cat([d5_ecg,e1],1)
        
        d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
#         print ("d6: ", d6.shape)
        
        d7_ecg = self.de7_ecg(d6_ecg)
#         print ("d7: ", d7.shape)
        d8_ecg = self.de8_ecg(d7_ecg)
 
        d9_ecg = self.de9_ecg(d8_ecg)
        
        return d9_ecg, out_4

    
class BRUnet_Encoder(nn.Module):
    def __init__(self, shape):
        super(BRUnet_Encoder, self).__init__()
        in_channels = shape[1]
        
        
        
        self.en1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 4, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        
        self.en5 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 3,padding = 1),
                                 IncBlock(512,512))
        
        self.en6 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(1024,1024))

        self.en7_p = nn.Sequential(nn.Conv1d(1024, 128, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(128,128))
        self.en8_p = nn.Sequential(nn.Conv1d(128, 64, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(64),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(64,64))

        self.en9_p = nn.Sequential(nn.Conv1d(64, 4, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(4),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(4,4))
        self.fc = nn.Linear(4,1)      
        
        #self.de1_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1, stride = 1),
        #                       nn.BatchNorm1d(512),
        #                       nn.LeakyReLU(0.2),
        #                        IncBlock(512,512))
        
        #self.de2_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1),
        #                       nn.BatchNorm1d(512),
        #                       nn.LeakyReLU(0.2),
        #                    nn.ConvTranspose1d(512,256,1),
        #                       IncBlock(256,256))
        
        #self.de3_ecg =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
        #                       nn.BatchNorm1d(256),
        #                       nn.LeakyReLU(0.2),
        #                        nn.ConvTranspose1d(256,128,4,stride = 2, padding =1),
        #                        IncBlock(128,128))
        
        #self.de4_ecg =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
        #                       nn.BatchNorm1d(128),
        #                       nn.LeakyReLU(0.2),
        #                        nn.ConvTranspose1d(128,64,3, stride = 2),
        #                        IncBlock(64,64))
        
        #self.de5_ecg =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
        #                       nn.BatchNorm1d(64),
        #                       nn.LeakyReLU(0.2),
        #                        nn.ConvTranspose1d(64,32,3, stride = 2),
        #                        IncBlock(32,32))
        
        #self.de6_ecg = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding =1),
        #                       nn.BatchNorm1d(32),
        #                       nn.LeakyReLU(0.2),
        #                        nn.ConvTranspose1d(32,16,3, stride = 2, padding = 0),
        #                        IncBlock(16,16))
                               
        #self.de7_ecg = nn.Sequential(nn.ConvTranspose1d(16,1,1,stride =1, padding = 0),
        #                        nn.LeakyReLU(0.2))
        #self.de8_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
        #                        nn.LeakyReLU(0.2))
        #self.de9_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
        #                        nn.LeakyReLU(0.2))
        
        

    def forward(self,x):
#         print (x.shape)
        e1 = self.en1(x)
#         print ("e1: ", e1.shape)
        
        e2 = self.en2(e1)
#         print ("e2: ", e2.shape)
# #         
        e3 = self.en3(e2)
#         print ("e3: ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("e4: ", e4.shape)
        
        e5 = self.en5(e4)
#         print ("e5 :", e5.shape)
        #import pdb;pdb.set_trace()
        e6 = self.en6(e5)

        out_1 = self.en7_p(e6)

        out_2 =  self.en8_p(out_1)
        #import pdb;pdb.set_trace()
        out_3 =  self.en9_p(out_2)
        out_4 = self.fc(torch.squeeze(out_3,-1))
        
#         print ("e6 :", e6.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
        #d1_ecg = self.de1_ecg(e6)
#         print ("d1: ",d1.shape)  


        #cat_ecg = torch.cat([d1_ecg,e5],1)
    

        #d2_ecg = self.de2_ecg(cat_ecg)
#         print ("d2: ", d2.shape)
        

        #cat_ecg = torch.cat([d2_ecg,e4],1)
        
        
        #d3_ecg = self.de3_ecg(cat_ecg)
        #d3_ecg = F.pad(d3_ecg, (0,1))
#         print ("d3: ",d3.shape)
        
        #cat_ecg = torch.cat([d3_ecg,e3],1)

        
        #d4_ecg = self.de4_ecg(cat_ecg)
        #d4_ecg = d4_ecg[:,:,:-1]
#         print ("d4: " ,d4.shape)

        #cat_ecg = torch.cat([d4_ecg,e2],1)
        
        
        #d5_ecg = self.de5_ecg(cat_ecg)
        #d5_ecg = d5_ecg[:,:,:-1]
#         print ("d5: ",d5.shape)
        
        #cat_ecg = torch.cat([d5_ecg,e1],1)
        
        #d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
#         print ("d6: ", d6.shape)
        
        #d7_ecg = self.de7_ecg(d6_ecg)
#         print ("d7: ", d7.shape)
        #d8_ecg = self.de8_ecg(d7_ecg)
 
        #d9_ecg = self.de9_ecg(d8_ecg)
        
        return out_4

class BRUnet_raw(nn.Module):
    def __init__(self, shape):
        super(BRUnet_raw, self).__init__()
        in_channels = shape[1]
        
        
        
        self.en1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 4, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        
        self.en5 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 3,padding = 1),
                                 IncBlock(512,512))
        
        self.en6 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(1024,1024))

        self.en7_p = nn.Sequential(nn.Conv1d(1024, 128, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(128,128))
        self.en8_p = nn.Sequential(nn.Conv1d(128, 64, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(64),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(64,64))

        self.en9_p = nn.Sequential(nn.Conv1d(64, 4, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(4),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(4,4))
        self.fc = nn.Linear(64,1)      
        
        #self.de1_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1, stride = 1),
        #                       nn.BatchNorm1d(512),
        #                       nn.LeakyReLU(0.2),
        #                        IncBlock(512,512))
        
        #self.de2_ecg = nn.Sequential(nn.ConvTranspose1d(1024,512,1),
        #                       nn.BatchNorm1d(512),
        #                       nn.LeakyReLU(0.2),
        #                    nn.ConvTranspose1d(512,256,1),
        #                       IncBlock(256,256))
        
        #self.de3_ecg =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
        #                       nn.BatchNorm1d(256),
        #                       nn.LeakyReLU(0.2),
        #                        nn.ConvTranspose1d(256,128,4,stride = 2, padding =1),
        #                        IncBlock(128,128))
        
        #self.de4_ecg =  nn.Sequential(nn.Conv1d(256,128,3, stride = 1, padding = 1),
        #                       nn.BatchNorm1d(128),
        #                       nn.LeakyReLU(0.2),
        #                        nn.ConvTranspose1d(128,64,3, stride = 2),
        #                        IncBlock(64,64))
        
        #self.de5_ecg =  nn.Sequential(nn.Conv1d(128,64,3, stride = 1, padding = 1),
        #                       nn.BatchNorm1d(64),
        #                       nn.LeakyReLU(0.2),
        #                        nn.ConvTranspose1d(64,32,3, stride = 2),
        #                        IncBlock(32,32))
        
        #self.de6_ecg = nn.Sequential(nn.Conv1d(64,32,3, stride = 1, padding =1),
        #                       nn.BatchNorm1d(32),
        #                       nn.LeakyReLU(0.2),
        #                        nn.ConvTranspose1d(32,16,3, stride = 2, padding = 0),
        #                        IncBlock(16,16))
                               
        #self.de7_ecg = nn.Sequential(nn.ConvTranspose1d(16,1,1,stride =1, padding = 0),
        #                        nn.LeakyReLU(0.2))
        #self.de8_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
        #                        nn.LeakyReLU(0.2))
        #self.de9_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
        #                        nn.LeakyReLU(0.2))
        
        

    def forward(self,x):
#         print (x.shape)
        e1 = self.en1(x)
#         print ("e1: ", e1.shape)
        
        e2 = self.en2(e1)
#         print ("e2: ", e2.shape)
# #         
        e3 = self.en3(e2)
#         print ("e3: ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("e4: ", e4.shape)
        
        e5 = self.en5(e4)
#         print ("e5 :", e5.shape)
        #import pdb;pdb.set_trace()
        e6 = self.en6(e5)

        out_1 = self.en7_p(e6)

        out_2 =  self.en8_p(out_1)
        #import pdb;pdb.set_trace()
        out_3 =  self.en9_p(out_2)
        #import pdb;pdb.set_trace() 
        out_4 = self.fc(out_3.view(-1,out_3.shape[1]*out_3.shape[2]))

        
#         print ("e6 :", e6.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
        #d1_ecg = self.de1_ecg(e6)
#         print ("d1: ",d1.shape)  


        #cat_ecg = torch.cat([d1_ecg,e5],1)
    

        #d2_ecg = self.de2_ecg(cat_ecg)
#         print ("d2: ", d2.shape)
        

        #cat_ecg = torch.cat([d2_ecg,e4],1)
        
        
        #d3_ecg = self.de3_ecg(cat_ecg)
        #d3_ecg = F.pad(d3_ecg, (0,1))
#         print ("d3: ",d3.shape)
        
        #cat_ecg = torch.cat([d3_ecg,e3],1)

        
        #d4_ecg = self.de4_ecg(cat_ecg)
        #d4_ecg = d4_ecg[:,:,:-1]
#         print ("d4: " ,d4.shape)

        #cat_ecg = torch.cat([d4_ecg,e2],1)
        
        
        #d5_ecg = self.de5_ecg(cat_ecg)
        #d5_ecg = d5_ecg[:,:,:-1]
#         print ("d5: ",d5.shape)
        
        #cat_ecg = torch.cat([d5_ecg,e1],1)
        
        #d6_ecg = self.de6_ecg(cat_ecg)[:,:,:-1]
#         print ("d6: ", d6.shape)
        
        #d7_ecg = self.de7_ecg(d6_ecg)
#         print ("d7: ", d7.shape)
        #d8_ecg = self.de8_ecg(d7_ecg)
 
        #d9_ecg = self.de9_ecg(d8_ecg)
        
        return out_4

class BRUnet_raw_Multi(nn.Module):
    def __init__(self, shape):
        super(BRUnet_raw_Multi, self).__init__()
        in_channels = shape[1]
        
        
        
        self.en1 = nn.Sequential(nn.Conv1d(in_channels, 32, 3, padding = 1), 
                                nn.BatchNorm1d(32),
                                nn.LeakyReLU(0.2),
                                nn.Conv1d(32, 32, 5, stride = 2, padding = 2),
                                IncBlock(32,32))
        
        self.en2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding = 1),
                                nn.BatchNorm1d(64),
                                nn.LeakyReLU(0.2),
                                 nn.Conv1d(64, 64, 5, stride = 2, padding = 2),
                                IncBlock(64,64))
        
              
        self.en3 = nn.Sequential(nn.Conv1d(64,128, 3, padding = 1),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(128, 128, 3, stride = 2, padding = 1),
                                IncBlock(128,128))
        
        self.en4 = nn.Sequential(nn.Conv1d(128,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 4, stride = 2, padding = 1),
                                IncBlock(256,256))
        
        
        self.en5 = nn.Sequential(nn.Conv1d(256,256, 3,padding = 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(256, 256, 3,padding = 1),
                                 IncBlock(256,256))
        
        self.en6 = nn.Sequential(nn.Conv1d(256,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 4, stride = 2, padding = 1),
                                IncBlock(512,512))
        
        
        
        self.en7 = nn.Sequential(nn.Conv1d(512,512, 3,padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(512, 512, 4, stride = 2, padding = 1),
                                IncBlock(512,512))
        
        self.en8 = nn.Sequential(nn.Conv1d(512,1024, 3,padding = 1),
                                 nn.BatchNorm1d(1024),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv1d(1024, 1024, 4, stride = 2, padding = 1),
                                IncBlock(1024,1024))
       

        self.en7_p = nn.Sequential(nn.Conv1d(512, 512, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(512,512))
        self.en8_p = nn.Sequential(nn.Conv1d(512, 512, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(512,512))

        self.en9_p = nn.Sequential(nn.Conv1d(512, 512, 4, stride = 2, padding = 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(0.2),
                                 IncBlock(512,512))
        self.fc = nn.Linear(512 * 8,1)      
        
        self.de1_ecg =  nn.Sequential(nn.Conv1d(1024,512,3, padding = 1),
                                nn.BatchNorm1d(512),
                                nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(512,512,4,stride = 2, padding =1),
                                IncBlock(512,512))

        self.de2_ecg =  nn.Sequential(nn.Conv1d(1024,512,3, padding = 1),
                                nn.BatchNorm1d(512),
                                nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(512,512,4,stride = 2, padding =1),
                                IncBlock(512,512))

        self.de3_ecg =  nn.Sequential(nn.Conv1d(1024,256,3, padding = 1),
                                nn.BatchNorm1d(256),
                                nn.LeakyReLU(0.2),
                                nn.ConvTranspose1d(256,256,4,stride = 2, padding =1),
                                IncBlock(256,256))

        self.de4_ecg =  nn.Sequential(nn.Conv1d(512,256,3, padding = 1),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(0.2),
                                        nn.ConvTranspose1d(256,256,3, padding =1),
                                        IncBlock(256,256))

        self.de5_ecg = nn.Sequential(nn.ConvTranspose1d(512,256,1),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(0.2),
                        nn.ConvTranspose1d(256,256,1),
                        IncBlock(256,256))
        
        self.de6_ecg = nn.Sequential(nn.ConvTranspose1d(256,128,1),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2),
                        nn.ConvTranspose1d(128,64,1),
                        IncBlock(64,64))
        
        self.de7_ecg = nn.Sequential(nn.ConvTranspose1d(64,8,1),
                        nn.BatchNorm1d(8),
                        nn.LeakyReLU(0.2),
                        nn.ConvTranspose1d(8,4,1),
                        IncBlock(4,4))

        self.de8_ecg = nn.Sequential(nn.ConvTranspose1d(4,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        self.de9_ecg = nn.Sequential(nn.ConvTranspose1d(1,1,1,stride =1, padding = 0),
                                nn.LeakyReLU(0.2))
        

    def forward(self,x):
#         print (x.shape)
        e1 = self.en1(x)
#         print ("e1: ", e1.shape)
        
        e2 = self.en2(e1)
#         print ("e2: ", e2.shape)
# #         
        e3 = self.en3(e2)
#         print ("e3: ",e3.shape)
        
        e4 = self.en4(e3)
#         print ("e4: ", e4.shape)
        
        e5 = self.en5(e4)
#         print ("e5 :", e5.shape)
        #import pdb;pdb.set_trace()
        e6 = self.en6(e5)
        #e5 = self.en5(e4)
#        print ("e5 :", e5.shape)
        
        #e6 = self.en6(e5)

        e7 = self.en7(e6)
        
        e8 = self.en8(e7)
        
        out_1 = self.en7_p(e6)

        out_2 =  self.en8_p(out_1)
        #import pdb;pdb.set_trace()
        out_3 =  self.en9_p(out_2)
        out_4 = self.fc(out_3.view(-1,out_3.shape[1] * out_3.shape[2]))
#         print ("e6 :", e6.shape)
# #         print ("-----------------------------------------------------------------------------")
#         #-----------------------------------------------------------------------------
        d1_ecg = self.de1_ecg(e8)
#         print ("d1: ",d1.shape)  

        cat_ecg = torch.cat([d1_ecg,e7],1)
    
        d2_ecg = self.de2_ecg(cat_ecg)
#         print ("d2: ", d2.shape)
        
        cat_ecg = torch.cat([d2_ecg,e6],1)
        
        
        d3_ecg = self.de3_ecg(cat_ecg)
        #d3_ecg = F.pad(d3_ecg, (0,1))
#         print ("d3: ",d3.shape)
        
        cat_ecg = torch.cat([d3_ecg,e5],1)

        
        d4_ecg = self.de4_ecg(cat_ecg)

        cat_ecg = torch.cat([d4_ecg,e4],1)
        
        
        d5_ecg = self.de5_ecg(cat_ecg)
        d6_ecg = self.de6_ecg(d5_ecg)
        d7_ecg = self.de7_ecg(d6_ecg)
        d8_ecg = self.de8_ecg(d7_ecg)
        d9_ecg = self.de9_ecg(d8_ecg)
        
        return d9_ecg, out_4
