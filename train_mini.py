
import torch, random, os, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)
class UNetMini(nn.Module):
    def __init__(self, chs=(3,32,64,128)):
        super().__init__()
        self.enc1 = DoubleConv(chs[0], chs[1])
        self.enc2 = DoubleConv(chs[1], chs[2])
        self.enc3 = DoubleConv(chs[2], chs[3])
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec2 = DoubleConv(chs[3], chs[2])
        self.up1 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec1 = DoubleConv(chs[2], chs[1])
        self.outc = nn.Conv2d(chs[1], 1, 1)
    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.outc(d1)
        return out

class CrackDataset(Dataset):
    def __init__(self, records, img_size=256):
        self.records = records; self.img_size=img_size
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r['image']).convert('RGB').resize((self.img_size,self.img_size))
        m = Image.open(r['mask']).convert('L').resize((self.img_size,self.img_size))
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(img)
        m = (transforms.ToTensor()(m)>0.5).float()
        return img, m

def prepare_records_from_folder(images_dir, masks_dir):
    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg')])
    recs=[]
    for p in imgs:
        stem = p.stem
        for ext in ('.png','.jpg','.jpeg'):
            cand = masks_dir / (stem+ext)
            if cand.exists():
                recs.append({'image':str(p),'mask':str(cand)})
                break
    return recs

def train_model(records, epochs=3, batch_size=2, lr=1e-4, out_path='trained_mini.pth', progress_cb=None):
    device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    ds = CrackDataset(records, img_size=256)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = UNetMini().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    history=[]
    total_steps = epochs * len(dl)
    step=0
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            xb=xb.to(device); yb=yb.to(device)
            out = model(xb)
            loss = bce(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            step+=1
            if progress_cb:
                progress_cb(min(step/total_steps,1.0))
        model.eval()
        ious=[]
        with torch.no_grad():
            for xb,yb in dl:
                xb=xb.to(device); out=model(xb); p = torch.sigmoid(out).cpu().numpy()
                for pp, yy in zip(p, yb.cpu().numpy()):
                    pb = (pp[0]>=0.5).astype('uint8'); yb0 = (yy[0]>=0.5).astype('uint8')
                    inter = (pb & yb0).sum(); union = (pb | yb0).sum()
                    iou = inter/union if union>0 else 1.0
                    ious.append(iou)
        mean_iou = sum(ious)/len(ious) if ious else 0.0
        history.append({'epoch':ep, 'mean_iou':mean_iou, 'loss':float(loss.cpu().item())})
    torch.save({'model_state':model.state_dict()}, out_path)
    return history

def infer_on_image(img_arr, model_path=None):
    import numpy as np, torch, cv2
    device = torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    H,W = img_arr.shape[0], img_arr.shape[1]
    model = UNetMini().to(device)
    if model_path and os.path.exists(str(model_path)):
        ck = torch.load(str(model_path), map_location=device)
        sd = ck.get('model_state', ck) if isinstance(ck, dict) else ck
        model.load_state_dict(sd, strict=False)
    model.eval()
    img = Image.fromarray(img_arr).convert('RGB').resize((256,256))
    x = transforms.ToTensor()(img)
    x = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(x).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    prob = torch.sigmoid(out).cpu().numpy()[0,0]
    mask = (prob>=0.5).astype('uint8')*255
    mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
    return mask
