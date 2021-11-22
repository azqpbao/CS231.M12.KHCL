import numpy as np
import cv2
import matplotlib.pyplot as plt

# thuc hien segmentation using opencv 
def seg(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [20,255,10]
    cv2.imwrite('result\\segUsingOpencv.jpg',img)

#tinh khoang cach giua 2 vector
def distance(vi,vj):    #su dung cong thuc tinh khoang cach giua 2 vector theo euclidean 
    
    dis = np.sqrt((vi[0]-vj[0])**2 + (vi[1]-vj[1])**2 + (vi[2]-vj[2])**2)
    return dis

def kmeans(features, k , num_iters = 20):
    """
        su dung thuat toan kmeans de gom nhom nhung features thanh k nhom
        hàm su dung thuat toan kmeans++ de tim ngau nhien k cluster center ban dau
        hieu qua hon
        su dung 1 so ham tu thu vien numpy giup kmeans chay nhanh hon: np.argmin ,np.argmax
        Args:
            features - la 1 list chua cac features - 1 vector 3 chieu (r,g,b) cua image
            k - so luong cluster centers
            num_iiters - so luong vong lap
        Returns:
            assignment - 1 list chua thong tin ve vector thu i (voi i thuoc features) thuoc nhom j (voi j thuoc k)
    """
    N, D = features.shape
    # khoi tao ngau nhien k cluster center ban dau
    centers = [features[np.random.choice(N,1)[0]]]
    for kk in range(1, k):
        D2 = np.array([min([distance(c, x)**2 for c in centers]) for x in features])
        probs = D2/D2.sum()
        i = np.argmax(probs)
        centers.append(features[i])
    print(centers)
    assignments = np.zeros(N)
    
    dis = np.zeros((N,k))
    for n in range(num_iters):  
        print("Vong lap thu: ", n)
        for i in range(N):
            for j in range(0,k):
                d = distance(centers[j], features[i])
                dis[i][j] = d
        assignments = np.argmin(dis, axis = 1)        
        #update new cluster
        centers = np.zeros((k,3))
        for i in range(N):
            clusterI = assignments[i]
            centers[clusterI][0] += features[i][0]
            centers[clusterI][1] += features[i][1]
            centers[clusterI][2] += features[i][2]
        for i in range(len(centers)):
            number = assignments.tolist().count(i)
            print("cluster = ", i, " number of point = ",number)
            centers[i][0] /= number
            centers[i][1] /= number
            centers[i][2] /= number
        print(centers)

    return assignments
 
#ham nay se giup keo dai do tuong phan - de xu ly nhung buc anh mo tro nen ro net hon
def constrastStretching(img):  
    
    min_Output = 0
    max_Output = 255
    
    min_Input = img.min()
    max_Input = img.max()
    Output = img
    
    for i in range (0,len(Output)):
        for j in range (0,len(Output[0])):
            Output[i][j] = (img[i][j]-min_Input)*(((max_Output-min_Output)/(max_Input-min_Input))+min_Output)     
    return Output

def color_position_features(img):
    """
        bieu dien moi pixel duoi dang 1 vector 3 chieu (red, green, blue)
        gia tri RGB cua 1 pixel anh xa thanh 1 feature vector
        Args:
            img - 1 buc anh mau, la 1 ma tran voi kich thuoc (H,W,C)
        Return:
            features - 1 list chua cac features vector voi kich thuoc (H*W,C)
    """
    H, W, C = img.shape
    img = img.astype(float)
    features = np.zeros((H*W, C))
    for x in range(0,H):
        for y in range(0,W):
            for k in range(C):
                features[W*x + y][k] = img[x][y][k]
    return features

def imgModify(imgSrc, segments, k):
    """
        su dung ket qua segments de thuc hien mot so ung dung tren anh
        Args:
            imgSrc - buc anh nguon
            segments - ket qua moi pixel se duoc thuoc nhom k
            k - so luong cluster 
    """
    img_gray = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY) 
    imgnew = np.copy(imgSrc)
    imgnew[:,:,0] = img_gray
    imgnew[:,:,1] = img_gray
    imgnew[:,:,2] = img_gray
    #lam anh trang den
    for i in range(k):
        finalImg = np.copy(imgSrc)
        finalImg[segments!=i] = imgnew[segments!=i]
        cv2.imwrite('result\\label%d.jpg'%(i), finalImg)
    
    
def evaluation(imgSrc, imgGT, segments, k):
    """
        su dung hinh anh duoc cat bang thu cong voi do chinh xac cao nhat
        de so sanh voi hinh anh duoc cat tu dong bang thuat toan k-means
        Args:
            imgSrc - hinh anh duoc cat bang thuat toan
            imgGT - hinh anh duoc cat thu cong - voi do chinh xac cao nhat
            segments  - phan cac pixel thuoc nhom i (voi i thuoc k)
            k - so luong cluster
        return:
            accurate - do chinh xac cua thuat toan k-means
    """
    accurate = np.zeros(k)
    imgSrc = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY)
    for i in range(k):
        imgBinary = np.copy(imgSrc)
        for x in range(imgSrc.shape[0]):
            for y in range(imgSrc.shape[1]):
                if segments[x][y] == i:
                    imgBinary[x][y] = 255
                else:
                    imgBinary[x][y] = 0
        TP = 0
        FP = 0
        FN = 0
        for x in range(imgGT.shape[0]):
            for y in range(imgGT.shape[1]):
                if (imgBinary[x][y] == 255 and imgGT[x][y] == 255):
                    TP += 1
                if (imgBinary[x][y] == 0 and imgGT[x][y] == 255):
                    FP += 1
                if (imgBinary[x][y] == 255 and imgGT[x][y] == 0):
                    FN += 1
        accurate[i] = (2*TP)/(2*TP+FP+FN)
    return max(accurate)
        
if __name__ == "__main__":
    # load hinh anh
    img = cv2.imread("test\case0\cat0.jpg")
    
    # contrast stretching mot buc anh
    img = constrastStretching(img)
    
    # H,W,C la kich thuoc cua ma tran image
    H, W, C = img.shape
    
    # features - tim cac feature vector cua mot image - moi pixel la 1 feature vector
    features = color_position_features(img)
    
    # goi thuat toan k-means
    k = 2
    assignments = kmeans(features, k, 7)
    segments = assignments.reshape(H,W)
    plt.imsave('result\segmentImg.png', segments)
    
    # thuc hien lam phan background cua buc anh thanh trang den
    imgModify(img,segments,k)
     
    # load hinh anh phan biet background voi foreground duoc thuc hien thu cong bang tay
    imgGT = cv2.imread("test\case0\cat0gt.jpg",0)
    
    #danh gia do chinh xac
    evals = evaluation(img, imgGT, segments, k)
    print('Accuracy: ', evals)
    
    #Image Segmentation using openCV
    seg(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
