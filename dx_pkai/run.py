import random

from protein import *
from util import *
from model import *
import warnings
warnings.filterwarnings('ignore')

def read_pkpdb():
    pka={}
    temp = np.array(pd.read_csv('D:\\pdb\\final_pka.csv', index_col=False))
    for i in temp:
        pka[i[0]+i[1]+str(i[2])+i[3]]=float(i[4])
    return pka

def read_pkad():
    pka = {}
    fd = pd.read_csv('D:\\pdb\\pkad\\PKAD2_DOWNLOAD.csv', index_col=False)
    fd = np.array(fd)
    # print(fd[:10])
    for i in fd:
        if '<' in i[4] or '>' in i[4] or '-' in i[4] or '~' in i[4]:
            # i[4]=i[4][1:]
            continue
        if i[1] == 'N-term' or i[1] == 'C-term':
            continue
        try:
            pka[i[0]+i[2]+str(i[3])+i[1]]=float(i[4])
        except:
            pass

    return pka

def run(device="cpu", threads=None):
    if threads:
        torch.set_num_threads(threads)
    #model = load_model(model_name, device)
    model=Net()
    #model = GAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_func = nn.MSELoss()

    min_mae = 100
    epoches = 1
    for epoch in range(epoches):
        predictions_tr = torch.Tensor()
        labels_tr = torch.Tensor()
        err = []
        losses = []
        file_num=0

        start = time.time()
        pdb_path = 'D:\\pdb\\pkPDB\\data\\'
        pssm_path = 'D:\\pdb\\pkPDB\\pssm\\'
        pka = read_pkpdb()

        files=os.listdir(pdb_path)
        random.shuffle(files)

        file_count=0
        file_length = len(files)
        for file_name in files:
            file_count+=1

            prot = Protein(pdb_path,file_name,pssm_path)
            res = prot.read_pssm()
            if res == -1:
                #print(file_name,'-1')
                continue

            try:
                prot.align_pssm()
                prot.align_pka(pka, file_name.replace('.pdb',''))
                prot.apply_cutoff()
                prot.creat_graph()

                optimizer.zero_grad()
                preds, pkas = prot.predict_pkas(model, device, loss_func, optimizer)
            except:
                #print(file_name)
                continue

            predictions_tr = torch.cat((predictions_tr, preds), 0)
            labels_tr = torch.cat((labels_tr, pkas), 0)

            file_num += 1
            if file_num >=50:
                loss = loss_func(predictions_tr, labels_tr)

                err.extend(abs(predictions_tr.view(-1).detach().numpy() - labels_tr.view(-1).detach().numpy()))

                loss.requires_grad_(True).backward()
                losses.append(loss.view(-1).detach().numpy())
                optimizer.step()

                predictions_tr = torch.Tensor()
                labels_tr = torch.Tensor()
                file_num = 0

                print('files:',file_count,'/',file_length,' time:', str(time.time() - start), '---MAE:', str(np.average(err)))
                start = time.time()
                err = []

        predictions_tr = torch.Tensor()
        labels_tr = torch.Tensor()

        print('test')
        model.eval()
        pkad_path = 'D:\\pdb\\pkad\\data\\'
        pkad_pssm_path = 'D:\\pdb\\pkad\\pssm\\'
        pkad = read_pkad()

        files = os.listdir(pkad_path)
        ave=[]
        for file_name in files:
            #print(file_name)
            prot_test = Protein(pkad_path,file_name,pkad_pssm_path)
            res = prot_test.read_pssm()
            if res == -1:
                #print(file_name, '-1')
                continue
            try:
                prot_test.align_pssm()
                prot_test.align_pka(pkad, file_name.replace('.pdb', ''))
                prot_test.apply_cutoff()
                prot_test.creat_graph()

                result=prot_test.test(model, device)
            except:
                #print(file_name)
                continue

            for i in result:
                if i[1]==-1:
                    #print(result)
                    continue
                ave.append(abs(i[0]-i[1]))
        mae = np.average(ave)
        print(mae)

        save_path = 'C:\\Users\\dx\\Documents\\dx_pkai_model_gat+20pssm\\dx_pkai_model_gat_pssm\\save_model\\'
        epoch_model_save_name = f'pka_net_epoch{epoch}_mae{mae:.5f}.pt'
        best_model_save_name = 'pka_net_best_mae.pt'
        torch.save(model.state_dict(), save_path + epoch_model_save_name)
        if min_mae > mae:
            torch.save(model.state_dict(), save_path + best_model_save_name)
            min_mae = mae


run(device="cpu", threads=None)