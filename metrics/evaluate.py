import sys
import glob
import rouge
import pandas as pd
import bleu
f1 = sys.argv[1] #decoded
f2 = sys.argv[2] #reference
ref = []
decoded = []

for i, j in zip(sorted(glob.glob(f1+'*.txt')),sorted(glob.glob(f2+'*.txt'))):
        ref_tex = ''
        dec_tex = ''
        for k in open(i).readlines():
                dec_tex = dec_tex + k.strip()
        #if len(dec_tex)
        for l in open(j).readlines():
                ref_tex = ref_tex + l.strip()
        ref.append(ref_tex)
        decoded.append(dec_tex)

d = {'decoded':decoded, 'reference': ref}
df = pd.DataFrame(d)
df.to_csv('analysis.csv',index = False)
bl = bleu.moses_multi_bleu(decoded,ref)
print(bl)
x = rouge.rouge(decoded,ref)
print(x)
print('%.2f\t%.2f\t%.2f\t%.2f'%(bl,x['rouge_1/f_score']*100,x['rouge_2/f_score']*100,x['rouge_l/f_score']*100))

