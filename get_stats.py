import json 
from tqdm import tqdm
import gzip
from urllib.parse import urlparse
import pickle
import pandas as pd


with open(home+'output/urls/userdict.dict','wb') as fo:
    inputdict=pickle.load(fo)


userdict={}

reliable=['repubblica.it','twitter.it','corriere.it','virgilio.it','gazzetta.it','upday.com','tgcom24.mediaset.it','libero.it','ilmessaggero.it','ilfattoquotidiano.it','fanpage.it',
          'leggo.it','lastampa.it','tuttomercatoweb.com','giallozafferano.it','sport.sky.it','ansa.it',
          'liberoquotidiano.it','ilgiornale.it','calciomercato.com','huffingtonpost.it','my-personaltrainer.it','bendingspoons.com','espresso.repubblica.it','ilmattino.it','italiaonline.it','ilsole24ore.com','donnamoderna.com','vanityfair.it','corrieredellosport.it','tuttosport.com','tpi.it','tg24.sky.it','ilgazzettino.it','ilpost.it','dailymotion.com','raiplay.it','mediasetplay.mediaset.it','adnkronos.com','notizie.tiscali.it','eurosport.it','tim.it','it.altervista.org','rainews.it','unionesarda.it','mymovies.it','affaritaliani.it','greenme.it','gds.it','smartworld.it','dagospia.com','la7.it','nostrofiglio.it','notizie.it','deejay.it','it.businessinsider.com','formulapassion.it','wired.it','deabyday.tv','ticketone.it','caffeinamagazine.it','milanofinanza.it','elle.com/it/','treccani.it','focus.it','corriereadriatico.it','grazia.it','ilbianconero.com','lacucinaitaliana.it','105.net','blog.cliomakeup.com','fcinternews.it','lanuovasardegna.it','alvolante.it','lagazzetta delmezzogiorno.it','zingarate.com','viamichelin.it','studenti.it','rockol.it','lasicilia.it','ilcentro.it','supereva.it','blitzquotidiano.it','cosmopolitan.it','gazzettadelsud.it','lettera43.it','ilgiornaledivicenza.it','ladyblitz.it','larena.it','wetransfer.com','prealpina.it','discoveryplus.it','quinews.net','filmtv.it','rai.it','quotidianodipuglia.it','iltempo.it','','skuola.net','ilmiolibro.it','marieclaire.com','','glamour.it','','vogue.it','termometropolitico.it','esquire.com','milleunadonna.it','mondadoristore.it']

df=pd.read_csv('fakeurls.csv')
df=df[df['domain_type']=='hoax']
unreliable=df['url'].tolist()

unreliable=['.'.join(urlparse(x).netloc.split('.')[-2:]) for x in unreliable]

reliable=[ '.'.join(x.split('.')[-2:]) for x in reliable if x!='']
reliable=[x for x in reliable if x not in unreliable]



for k,v in tqdm(inputdict.items()):

	rel_user=[x for x in v if '.'.join(urlparse(x).netloc.split('.')[-2:]) in reliable]
	unrel_user=[x for x in v if '.'.join(urlparse(x).netloc.split('.')[-2:]) in unreliable]
	tot=[x for x in v if '.'.join(urlparse(x).netloc.split('.')[-2:]) not in reliable+unreliable]

	userdict[k]={'num_rel':len(rel_user), 'num_unrel':len(unrel_user), 'lost':len(tot)}

with open(home+'output/urls/user_quantities.dict','wb') as fo:
    pickle.dump(userdict, fo)




