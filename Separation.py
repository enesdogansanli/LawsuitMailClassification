Lawlist = ['Ceza_Muhakemesi_Kanunu','Hukuk_Muhakemeleri_Kanunu','Icra_Iflas_Kanunu','Turk_Borclar_Kanunu','Turk_Ceza_Kanunu','Turk_Medeni_Kanunu','Turk_Ticaret_Kanunu']

documents = {'Ceza_Muhakemesi_Kanunu': (330,'Madde'),
            'Hukuk_Muhakemeleri_Kanunu': (440,'MADDE'),
            'Icra_Iflas_Kanunu':(360,'Madde'),
            'Turk_Borclar_Kanunu':(630,'MADDE'),
            'Turk_Ceza_Kanunu':(340,'Madde'),
            'Turk_Medeni_Kanunu':(998,'Madde'),
            'Turk_Ticaret_Kanunu':(998,'MADDE')}

for i in range(len(Lawlist)):
    with open('sourcedata/{}.txt'.format(Lawlist[i]),"r",encoding='utf8') as d:
        metin = d.read()
        ayirma = metin.split(documents[Lawlist[i]][1])
        for j in range(1,documents[Lawlist[i]][0]):
            if j <10:
                with open('data/{}/00{}.txt'.format(Lawlist[i],j),'w',encoding='utf8') as t:
                    t.write(ayirma[j])
            elif 9<j<100:
                with open('data/{}/0{}.txt'.format(Lawlist[i],j),'w',encoding='utf8') as t:
                    t.write(ayirma[j])
            else:
                with open('data/{}/{}.txt'.format(Lawlist[i],j),'w',encoding='utf8') as t:
                    t.write(ayirma[j])