# Use PyQtGraph to visualize scrolling point clouds

from multiprocessing import Process, Value
from queue import Empty
import time
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore

from vme_research.messaging.shared_ndarray import SharedNDArrayPipe


class PointCloud(Process):
    def __init__(
        self,
        stop,
        pub_sub: SharedNDArrayPipe,
        name="Point Cloud",
        max_vis_points=300000,
        x_scale=640,
        y_scale=480,
        z_scale=1,
        mode="scrolling",
        scrolling_t_scale=1.0,
        max_buffer_points=300000,
    ):
        super(PointCloud, self).__init__(daemon=True)
        self.stop = stop
        self.pub_sub = pub_sub
        self.name = name
        self.max_vis_points = max_vis_points
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.z_scale = z_scale
        self.mode = mode
        self.scrolling_t_scale = scrolling_t_scale
        self.max_buffer_points = max_buffer_points
        self.t_last = time.time()

    def run(self):
        try:
            self.run_live()
        except KeyboardInterrupt:
            pass

    def run_live(self):
        app = pg.mkQApp(self.name)
        w = gl.GLViewWidget()
        w.show()
        w.setWindowTitle(self.name)
        w.setCameraPosition(distance=20)

        g = gl.GLGridItem()
        w.addItem(g)

        pos3 = np.zeros((self.max_vis_points, 3))
        sp3 = gl.GLScatterPlotItem(pos=pos3, color=(1,1,1,.3), size=0.03, pxMode=False)
        w.addItem(sp3)

        self.events_t_all = None
        self.events_xy_all = None
        def update():
            data = self.pub_sub.get(N=-1)
            if data is None:
                return

            new_events, = data
            events_t = new_events[:, 0]
            events_xy = new_events[:, 1:3]

            if self.events_t_all is None:
                self.events_t_all = events_t
                self.events_xy_all = events_xy

            self.events_t_all  = np.concatenate((self.events_t_all,  events_t))
            self.events_xy_all = np.concatenate((self.events_xy_all, events_xy))

            if self.events_t_all.shape[0] > 0:

                if self.mode == "scrolling":
                    t0 = self.events_t_all[-1] - self.z_scale
                    keep = self.events_t_all > t0
                    self.events_t_all  = self.events_t_all [keep]
                    self.events_xy_all = self.events_xy_all[keep]
                elif self.mode == "drop":
                    if self.events_t_all.shape[0] > self.max_buffer_points:
                        self.events_t_all = self.events_t_all[-self.max_buffer_points:]
                        self.events_xy_all = self.events_xy_all[-self.max_buffer_points:, :]

                if self.events_t_all.shape[0] > 0:
                    if self.mode == "scrolling":
                        events_t_vis = 10*(((self.events_t_all - np.min(self.events_t_all)) / self.z_scale) - 0.5)
                    else:
                        events_t_vis = 10*((self.events_t_all / self.z_scale) - 0.5)
                    events_x_vis =   10*(self.events_xy_all[:, 0] / self.x_scale) - 5
                    events_y_vis =  -10*(self.events_xy_all[:, 1] / self.x_scale) + (10 * self.y_scale / self.x_scale)

                    # Swap some axis to make the visualization nicer
                    max_events = min(self.max_vis_points, events_t_vis.shape[0])
                    if max_events < events_t_vis.shape[0]:
                        event_indices = np.linspace(0, events_t_vis.shape[0]-1, num=max_events, dtype=np.int32)

                        pos3[:max_events, 0] = events_t_vis[event_indices]
                        pos3[:max_events, 1] = events_x_vis[event_indices]
                        pos3[:max_events, 2] = events_y_vis[event_indices]
                        pos3[max_events:, ...] = 0
                    else:
                        pos3[:max_events, 0] = events_t_vis[:max_events]
                        pos3[:max_events, 1] = events_x_vis[:max_events]
                        pos3[:max_events, 2] = events_y_vis[:max_events]
                        pos3[max_events:, ...] = 0

                    sp3.setData(pos=pos3)

        t = QtCore.QTimer()
        t.timeout.connect(update)
        t.start(1)
        pg.exec()

if __name__ == '__main__':
    stop = Value('i', 0)
    pub_sub = SharedNDArrayPipe(sample_data=(np.zeros((3,)),), max_messages=10000000)

    cloud = PointCloud(stop, pub_sub, mode='drop')
    cloud.start()
   
    t_start = time.time()
    t_last = time.time()
    while time.time() - t_start < 10.0:
        dt = time.time() - t_last
        N_e = int(1e5 * dt)
        events_t = np.linspace(t_last-t_start, t_last + dt - t_start, num=N_e)
        events_xy = np.random.randint(0, 480, (N_e, 2))
        events_p = np.zeros((N_e,))
        t_last = time.time()

        stacked = np.hstack((events_t[:, None], events_xy))
    
        time.sleep(0.0166)
        pub_sub.pub((stacked,))
