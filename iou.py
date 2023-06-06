def IoU(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
  """
  x1,y1 - coordinate of the left upper corner
  x2, y2 - coordinates of the right lower corner-
  box1: [x1,y1,x2,y2] coordinates of the ground truth box
  box2: [x1,y1,x2,y2] coordinates of the predicted box
  return: IoU between two boxes if they are overlapping, 0 otherwise 
  """
 # determine coordinates of the intersection
  left_x = max(box2[0], box1[0])
  top_y = max(box2[1], box1[1])
  right_x = min(box2[2], box1[2])
  bottom_y = min(box2[3], box1[3])
  
  # compute intersection area
  interArea = abs(max((right_x - left_x), 0) * max((bottom_y - top_y), 0))

  # compute the area of both the prediction and ground-truth
  ground_area = abs((box1[0] - box1[2]) * (box1[1] - box1[3]))
  predicted_area = abs((box2[0] - box2[2]) * (box2[1] - box2[3]))

  # compute the IoU
  return interArea / (ground_area + predicted_area - interArea)