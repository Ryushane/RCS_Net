k = 0
with open('train.txt','w') as f:
    for i in range(15):
        if(k==5):
            k = 1
        else:
            k += 1
        #filename = '%2d' %i + '.png'
        #f.write('%2d' %i + '.png' + str(k))
        f.write('%02d'%(i+1)+'.png'+' '+str(k)+'\n')
