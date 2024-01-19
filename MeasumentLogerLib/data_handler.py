import os

def get_frames_path(*args):
    frames_path = r'D:\Documentos_HD\Codes\text_reader\frames'

    num_frames = len(os.listdir(frames_path))
    if num_frames == 0:
        return []

    if len(args) != 0 and (len(set([type(arg) for arg in args])) != 1 or str(type(args[0])) != '<class \'int\'>'):
        print("<! Invalid argument: argument must be of type 'int' !>")
        exit(1)

    if len(args) > 3:
        print("<! Invalid argument: max number of arguments is 3 !>")
        exit(1)
    elif len(args) == 0:
        init_ind = 0
        end_ind = num_frames
        step = 1
    elif len(args) == 1:
        init_ind = 0
        end_ind = args[0]
        step = 1
    elif len(args) == 2:
        init_ind = args[0]
        end_ind = args[1]
        step = 1
    elif len(args) == 3:
        init_ind = args[0]
        end_ind = args[1]
        step = args[2]

    return [os.path.join(frames_path, file)
            for file in [f"{num}.jpg" for num in range(init_ind,end_ind,step)]
            if os.path.isfile(os.path.join(frames_path, file))]


def get_single_frame_path(ind=0):
    return f'../frames/{ind}.jpg'


if __name__ == '__main__':
    #print(get_frames_path(100, 122, 4))
    print(get_frames_path(6))
    #print(get_single_frame_path(3))
