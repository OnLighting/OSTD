import json
# 把任意一个图可视化一下,看 OBB 4 顶点和 bbox 中心的关系import json, math
d = json.load(open(r'C:\Users\23563\Desktop\揭榜挂帅\add_data\COCO_Format\ShipRSImageNet_train_rotatedbox_level_3.json','r',encoding='utf-8'))
img = d['images'][0]
for a in d['annotations'][:1]:
    if a['image_id'] == img['id']:
        seg = a['segmentation'][0] # [x1,y1,x2,y2,x3,y3,x4,y4]
        bbox5 = a['bbox']            # [x,y,w,h,angle]
          # 两个猜测:                                                                                                                                                                                    
          # 猜测 A: bbox = (cx, cy, w, h, angle_rad),segmentation 是 4 角 polygon                                                                                                                        
          # 猜测 B: bbox = (x_top_left, y_top_left, w, h, angle_rad),segmentation 是 4 角 polygon                                                                                                        
        poly_cx = sum(seg[0::2]) / 4
        poly_cy = sum(seg[1::2]) / 4
        print('bbox:', bbox5)
        print('polygon 顶点:', list(zip(seg[0::2], seg[1::2])))
        print('polygon center:', (poly_cx, poly_cy))
        print('bbox[0:2] (= (x,y)):', bbox5[:2])
        print('==> 判断: bbox[0:2] 是中心还是左上?')