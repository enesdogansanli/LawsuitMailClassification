



LABELS = ['Ceza_Muhakemesi_Kanunu','Hukuk_Muhakemeleri_Kanunu','Icra_Iflas_Kanunu','Turk_Borclar_Kanunu','Turk_Ceza_Kanunu','Turk_Medeni_Kanunu','Turk_Ticaret_Kanunu']


def create_data_set():
    with open('data.txt','w',encoding='utf8') as outfile:
        for label in LABELS:
            dir = '%s/%s' % (BASE_DIR,label)
            for filename in os.listdir(dir):
                fullfilname = '%s/%s' % (dir,filename)
                print(fullfilname)
                with open (fullfilname,'rb') as file:
                    text = file.read().decode(errors='replace').replace('\n', '')
                    outfile.write('%s\t%s\t%s\n' % (label,filename, text))







if __name__=='__main__':
    create_data_set()