#!/usr/bin/env python3


from tkinter import font as tkfont
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog, simpledialog
from tkinter import ttk
import numpy as np

from PIL import Image, ImageTk

import cv2
import ctypes
import time

import libMastaba
import json


# Make all windows map pixels 1:1 to screen resolution
awareness = ctypes.c_int()
errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)


large_mask = None
large_tmpl = None


def overlay_mask( image, mask, alpha = 0 ):
    inv_mask = cv2.bitwise_not( mask )
    b, g, r = cv2.split( cv2.cvtColor(image, cv2.COLOR_GRAY2BGR ) )
    
    b = cv2.bitwise_and( b, inv_mask )
    g = cv2.bitwise_and( g, inv_mask )
    r = cv2.bitwise_and( r, mask )
    return cv2.merge([b, r, g])


def show_mask_edit():
    #tmp = cv2.merge([large_tmpl // 2,  large_mask, large_tmpl // 2])
    tmp = overlay_mask( large_tmpl, large_mask )
    cv2.imshow( "Mask", tmp)
    cv2.waitKey(1)
    #cv2.imshow( "large_tmpl", large_tmpl)


def mouse_apply_button( x, y, button ):
    global large_mask

    if button == 0:
        cv2.circle( large_mask, (x, y), 20, 255, -1 )
    else:
        cv2.circle( large_mask, (x, y), 20, 0, -1 )

    show_mask_edit()

_mouse_down_flag = False
_mouse_down_button = 0

def mouse_handler( event, x, y, flags, param ):
    global _mouse_down_flag, _mouse_down_button

    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_down_flag = True
        _mouse_down_button = 0
    
    if event == cv2.EVENT_RBUTTONDOWN:
        _mouse_down_flag = True
        _mouse_down_button = 1
    

    if event == cv2.EVENT_MOUSEMOVE:
        if _mouse_down_flag:
            mouse_apply_button( x, y, _mouse_down_button )
    
    if event == cv2.EVENT_LBUTTONUP:
        _mouse_down_flag = False
    
    if event == cv2.EVENT_RBUTTONUP:
        _mouse_down_flag = False
    
    
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse_apply_button( x, y, 0)
    
    if event == cv2.EVENT_RBUTTONDBLCLK:
        mouse_apply_button( x, y, 1)



class AppWindow( object ):
    def __init__( self ):
        self.window = Tk()
        

        self.ncc = libMastaba.NCCMatcher()
        self.original_image = None
        
        #self.window.option_add( "*Font", "B612" )
        
        self.window.title("Mastaba template helper")
        
        bgcolour = "#dddddd"
        self.window.config( background = bgcolour )
        self.bgcolour = bgcolour


        row = 0
        btn0 = Button(self.window, text="Load image", command = self.load_image )
        btn0.grid(column=0, row=row, pady=10, padx=10)
        row += 1
    

        btn2 = Button(self.window, text="Load template", command = self.load_template )
        btn2.grid(column=0, row=row, pady=10, padx=10)
        row += 1

        #btn2 = Button(self.window, text="Perform search", command = self.placeholder )
        #btn2.grid(column=0, row=row, pady=10, padx=10)
        #row += 1

        ttk.Separator(self.window, orient="horizontal").grid( row = row , column = 0, columnspan=3, sticky = "ew")
        row += 1

        btn1 = Button(self.window, text="Select template", command = self.select_template )
        btn1.grid(column=0, row=row, pady=10, padx=10)
        
        btn1b = Button(self.window, text="Edit mask", command = self.edit_mask )
        btn1b.grid(column=1, row=row, pady=10, padx=10)
        
        btn1c = Button(self.window, text="Save mask", command = self.save_mask )
        btn1c.grid(column=2, row=row, pady=10, padx=10)
        
        
        row += 1



        #Label( self.window, text = "Template name:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        #self.e_template_name = Entry( self.window )
        #self.e_template_name.grid( column = 1, row = row, pady = 10, padx = 10 )
        #row += 1

        Label( self.window, text = "Template max level:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        self.e_template_max_level = Entry( self.window )
        self.e_template_max_level.grid( column = 1, row = row, pady = 10, padx = 10 )
        self.e_template_max_level.insert(END, "4")
        row += 1
        
        #self.e_template_limited_rotation_var = IntVar()
        #self.e_template_limited_rotation = Checkbutton(self.window, text = "Limited rotation", var = self.e_template_limited_rotation_var, onvalue=1, offvalue = 0, command = self.placeholder, bg = self.bgcolour)
        #self.e_template_limited_rotation.grid( column = 1, row = row, pady = 2, padx = 10, sticky = "w" )
        #self.e_template_limited_rotation.deselect()
        Label( self.window, text = "Rotation granularity:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        self.e_template_rotation_var = StringVar()
        rot_labels = ["Extra coarse", "Coarse", "Fine", "Very fine", "Ultra fine"]
        self.e_template_rotation = OptionMenu(self.window, self.e_template_rotation_var, *rot_labels )
        self.e_template_rotation.grid( column = 1, row = row, pady = 2, padx = 10, sticky = "w" )
        self.e_template_rotation_var.set("Fine")
        row += 1

        #Label( self.window, text = "Template min level:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        #self.e_template_min_level = Entry( self.window )
        #self.e_template_min_level.grid( column = 1, row = row, pady = 10, padx = 10 )
        #self.e_template_min_level.insert(END, "0")
        #row += 1

        self.l_is_computed = Label( self.window, text = "Template not computed!", bg = self.bgcolour )
        self.l_is_computed.grid( column = 1, row = row, pady = 10, padx = 10)


        btn3 = Button(self.window, text="Compute template", command = self.compute_template )
        btn3.grid(column=0, row=row, pady=10, padx=10)
        row += 1


        ttk.Separator(self.window, orient="horizontal").grid( row = row , column = 0, columnspan=3, sticky = "ew")
        row += 1

        Label( self.window, text = "Search max level:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        self.e_search_max_level = Entry( self.window )
        self.e_search_max_level.grid( column = 1, row = row, pady = 10, padx = 10 )
        self.e_search_max_level.insert(END, "4")
        row += 1

        Label( self.window, text = "Search min level:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        self.e_search_min_level = Entry( self.window )
        self.e_search_min_level.grid( column = 1, row = row, pady = 10, padx = 10 )
        self.e_search_min_level.insert(END, "0")
        row += 1

        Label( self.window, text = "Search correlation:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        self.e_search_correlation = Entry( self.window )
        self.e_search_correlation.grid( column = 1, row = row, pady = 10, padx = 10 )
        self.e_search_correlation.insert(END, "0.85")
        row += 1

        Label( self.window, text = "Search correlation, verify:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        self.e_search_correlation_verify = Entry( self.window )
        self.e_search_correlation_verify.grid( column = 1, row = row, pady = 10, padx = 10 )
        self.e_search_correlation_verify.insert(END, "0.85")
        row += 1

        Label( self.window, text = "Refine correlation:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        self.e_refine_correlation = Entry( self.window )
        self.e_refine_correlation.grid( column = 1, row = row, pady = 10, padx = 10 )
        self.e_refine_correlation.insert(END, "0.95")
        row += 1

        if False:
            Label( self.window, text = "Start angle:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
            self.e_start_angle = Entry( self.window )
            self.e_start_angle.grid( column = 1, row = row, pady = 10, padx = 10 )
            self.e_start_angle.insert(END, "-180")
            row += 1

            Label( self.window, text = "Stop angle:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
            self.e_stop_angle = Entry( self.window )
            self.e_stop_angle.grid( column = 1, row = row, pady = 10, padx = 10 )
            self.e_stop_angle.insert(END, "180")
            row += 1


        Label( self.window, text = "Search ROI:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        self.e_search_roi = Entry( self.window )
        self.e_search_roi.grid( column = 1, row = row, pady = 10, padx = 10 )
        self.e_search_roi.insert(END, "")
        btn4 = Button(self.window, text="Select ROI", command = self.select_roi )
        btn4.grid(column=2, row=row, pady=10, padx=10)
        row += 1

        Label( self.window, text = "NMS radius:", bg = self.bgcolour ).grid( column = 0, row = row, pady = 10, padx = 10 )
        self.e_nms_radius = Entry( self.window )
        self.e_nms_radius.grid( column = 1, row = row, pady = 10, padx = 10 )
        self.e_nms_radius.insert(END, "150")
        row += 1

        self.e_extra_padding_var = IntVar()
        self.e_extra_padding = Checkbutton(self.window, text = "Image padding", var = self.e_extra_padding_var, onvalue=1, offvalue = 0, command = self.placeholder, bg = self.bgcolour)
        self.e_extra_padding.grid( column = 1, row = row, pady = 2, padx = 10, sticky = "w" )
        self.e_extra_padding.select()
        row += 1

        self.e_template_padding_var = IntVar()
        self.e_template_padding = Checkbutton(self.window, text = "Template padding", var = self.e_template_padding_var, onvalue=1, offvalue = 0, command = self.placeholder, bg = self.bgcolour)
        self.e_template_padding.grid( column = 1, row = row, pady = 2, padx = 10, sticky = "w" )
        self.e_template_padding.select()
        row += 1

        self.e_use_mask_var = IntVar()
        self.e_use_mask = Checkbutton(self.window, text = "Use template mask", var = self.e_use_mask_var, onvalue=1, offvalue = 0, command = self.placeholder, bg = self.bgcolour)
        self.e_use_mask.grid( column = 1, row = row, pady = 2, padx = 10, sticky = "w" )
        self.e_use_mask.select()
        row += 1

        self.e_use_cross_corr_var = IntVar()
        self.e_use_cross_corr = Checkbutton(self.window, text = "Use strict correlation", var = self.e_use_cross_corr_var, onvalue=1, offvalue = 0, command = self.placeholder, bg = self.bgcolour)
        self.e_use_cross_corr.grid( column = 1, row = row, pady = 2, padx = 10, sticky = "w" )
        self.e_use_cross_corr.deselect()
        row += 1



        self.e_use_debug_var = IntVar()
        self.e_use_debug = Checkbutton(self.window, text = "Debug displays", var = self.e_use_debug_var, onvalue=1, offvalue = 0, command = self.placeholder, bg = self.bgcolour)
        self.e_use_debug.grid( column = 1, row = row, pady = 2, padx = 10, sticky = "w" )
        self.e_use_debug.select()
        

        btn4 = Button(self.window, text="Perform search", command = self.perform_search )
        btn4.grid(column=0, row=row, pady=10, padx=10)
        row += 1

        btn5 = Button(self.window, text="Save template & params", command = self.save_template )
        btn5.grid(column=0, row=row, pady=10, padx=10)
        row += 1


        menubar = Menu( self.window )
        filemenu = Menu( menubar, tearoff = 0 )
        filemenu.add_command( label = "Exit", command = self.window.quit)
        menubar.add_cascade( label = "File", menu = filemenu )
        
        self.window.config( menu = menubar )
        
        self.periodic_update()

    def placeholder( self ):
        pass


    def load_image( self ):
        #fn = filedialog.askopenfilename( filetypes = ( ('JPEG image', '*.jpg'), ('PNG image', '*.png'), ('All files', '*.*') ) )
        fn = filedialog.askopenfilename( filetypes = ( ('All files', '*.*'), ('JPEG image', '*.jpg'), ('PNG image', '*.png') ) )
        if fn is not None:
            if len(fn) > 0:
                self.original_image = cv2.imread( fn )
                self.bw_image = cv2.cvtColor( self.original_image, cv2.COLOR_BGR2GRAY )

                cv2.imshow( "Image", self.original_image )

    def select_template( self ):
        tmpimg = self.original_image.copy()
        cv2.rectangle(tmpimg, (0,0), (tmpimg.shape[1]-1, tmpimg.shape[0]-1), (0,255,0), 3 )
        roibb = cv2.selectROI("Image", tmpimg, showCrosshair=True, fromCenter=False)
        #self.template = self.original_image[]
        (x,y,w,h) = roibb
        if w > 10 and h > 10:
            self.template_image = self.bw_image[ y:y+h, x:x+w ]
            cv2.imshow( "Template", self.template_image )
        cv2.imshow( "Image", self.original_image )
        self.l_is_computed.configure(text = "Template not computed!")

        self.template_mask = np.ones( self.template_image.shape, dtype = np.uint8 )  * 255


    def select_roi( self ):
        tmpimg = self.original_image.copy()
        cv2.rectangle(tmpimg, (0,0), (tmpimg.shape[1]-1, tmpimg.shape[0]-1), (0,255,0), 3 )
        roibb = cv2.selectROI("Image", tmpimg, showCrosshair=True, fromCenter=False)
        #self.template = self.original_image[]
        (x,y,w,h) = roibb
        self.e_search_roi.delete(0, END)
        self.e_search_roi.insert(END, "%s" % json.dumps( roibb ))


    



    def compute_template( self ):
        max_level = int( self.e_template_max_level.get() )

        #limited = self.e_template_limited_rotation_var.get() == 1
        #min_level = int( self.e_template_min_level.get() )
        rot_level = self.e_template_rotation_var.get()


        print( "Precomputing template....")        
        #if not limited:
        #    tmpl = libMastaba.Precompute.generate_template( self.template_image, mask=self.template_mask, Nlevels=max_level )
        #else:
        #    tmpl = libMastaba.Precompute.generate_template( self.template_image, delta_angle = 45, mask=self.template_mask, Nlevels=max_level )
        #rot_labels = ["Extra coarse", "Coarse", "Fine", "Very fine", "Ultra fine"]
        
        if rot_level == "Extra coarse":
            tmpl = libMastaba.Precompute.generate_template( self.template_image, delta_angle = 45, mask=self.template_mask, Nlevels=max_level )
        elif rot_level == "Coarse":
            tmpl = libMastaba.Precompute.generate_template( self.template_image, coarseness_factor=15.0, mask=self.template_mask, Nlevels=max_level )
        elif rot_level == "Fine":
            tmpl = libMastaba.Precompute.generate_template( self.template_image, coarseness_factor=8.0, mask=self.template_mask, Nlevels=max_level )
        elif rot_level == "Very fine":
            tmpl = libMastaba.Precompute.generate_template( self.template_image, coarseness_factor=4.0, mask=self.template_mask, Nlevels=max_level )
        elif rot_level == "Ultra fine":
            tmpl = libMastaba.Precompute.generate_template( self.template_image, coarseness_factor=1.0, mask=self.template_mask, Nlevels=max_level )
        
        
        self.tmpl = tmpl
        
        print( "Loading template....")
        self.ncc.add_template_noload( "t", tmpl, numLevels = max_level )

        print( "Done.")
        self.l_is_computed.configure(text = "Template computed.")



    def perform_search( self ):
        searchLevel = int( self.e_search_max_level.get() )
        refineLevel = int( self.e_search_min_level.get() )
        searchCorrelation = float( self.e_search_correlation.get() )
        refineCorrelation = float( self.e_refine_correlation.get() )
        
        nmsRadius = float( self.e_nms_radius.get() )

        searchCorrelationVerify = float( self.e_search_correlation_verify.get() )
        

        #startAngle = float( self.e_start_angle.get() )
        #stopAngle = float( self.e_stop_angle.get() )
        

        extraPadding = self.e_extra_padding_var.get() == 1
        useTemplatePadding = self.e_template_padding_var.get() == 1
        useMaskForMatching = self.e_use_mask_var.get() == 1
        
        crosscorr = self.e_use_cross_corr_var.get() == 1

        use_debug = self.e_use_debug_var.get() == 1


        self.ncc.configure( "t", searchLevel = searchLevel, refineLevel = refineLevel )
        self.ncc.configure( "t", searchCorrelation = searchCorrelation, refineCorrelation = refineCorrelation )
        self.ncc.configure( "t", nmsRadius = nmsRadius )
        self.ncc.configure( "t", extraPadding = extraPadding, useTemplatePadding = useTemplatePadding, useMaskForMatching = useMaskForMatching )
        #self.ncc.configure( "t", startAngle = startAngle, stopAngle = stopAngle )

        self.ncc.configure( "t", searchCorrelationVerify = searchCorrelationVerify )

        self.ncc.configure( "t", drawResults = True )

        if crosscorr:
            self.ncc.configure( "t", corrMethod = cv2.TM_CCOEFF_NORMED )
        else:
            self.ncc.configure( "t", corrMethod = cv2.TM_CCORR_NORMED )


        roi = self.e_search_roi.get().strip()
        if len( roi ) > 0:
            roi = json.loads(roi)
            self.ncc.configure ("t", ROI = roi )
        else:
            if "ROI" in self.ncc.parameters["t"]:
                del self.ncc.parameters["t"]["ROI"]

        self.params = self.ncc.parameters["t"]


        self.ncc.configure( "t", debug = use_debug )

        t0 = time.time()
        results, _nccimg = self.ncc.detect( "t", self.original_image.copy() )
        t1 = time.time()
        print( "It took: %.2f ms" % ((t1-t0)*1000))

        cv2.imshow( "Image", _nccimg )


        print( "" )
        print( "Results:")
        for res in results:
            print( "\t", res )
        print( "" )


    def save_template( self ):
        fn = filedialog.asksaveasfilename( defaultextension=".npz", filetypes=(("Numpy data file", "*.npz"),("All Files", "*.*") ) )
        
        if "debug" in self.params:
            del self.params["debug"]


        self.tmpl["params"] = self.params

        
        print( "Saving to disk...")
        np.savez_compressed( fn, **self.tmpl )
        print( "Saved.")


    def load_template( self ):
        fn = filedialog.askopenfilename( filetypes = (("Numpy data file", "*.npz"),("All Files", "*.*") ) )
        
        print( "Loading template...")
        self.ncc.add_template("t", fn )
        print( "Template loaded." )

        #print( self.ncc.templates["t"]["params"])
        params = self.ncc.templates["t"]["params"]

        if "searchLevel" in params:
            self.e_search_max_level.delete(0, END)
            self.e_search_max_level.insert(END, "%i" % params["searchLevel"] )
        
        if "refineLevel" in params:
            self.e_search_min_level.delete(0, END)
            self.e_search_min_level.insert(END, "%i" % params["refineLevel"] )
        
        if "searchCorrelation" in params:
            self.e_search_correlation.delete(0, END)
            self.e_search_correlation.insert(END, "%.2f" % params["searchCorrelation"] )
        
        if "searchCorrelationVerify" in params:
            self.e_search_correlation_verify.delete(0, END)
            self.e_search_correlation_verify.insert(END, "%.2f" % params["searchCorrelationVerify"] )
        
        if "refineCorrelation" in params:
            self.e_refine_correlation.delete(0, END)
            self.e_refine_correlation.insert(END, "%.2f" % params["refineCorrelation"] )
        
        if "nmsRadius" in params:
            self.e_nms_radius.delete(0, END)
            self.e_nms_radius.insert(END, "%i" % params["nmsRadius"] )
        
        if "extraPadding" in params:
            if params["extraPadding"]:
                self.e_extra_padding.select()
            else:
                self.e_extra_padding.deselect()

        if "useTemplatePadding" in params:
            if params["useTemplatePadding"]:
                self.e_template_padding.select()
            else:
                self.e_template_padding.deselect()
        
        if "useMaskForMatching" in params:
            if params["useMaskForMatching"]:
                self.e_use_mask.select()
            else:
                self.e_use_mask.deselect()
        
        if "corrMethod" in params:
            if params["corrMethod"] != 3:
                self.e_use_cross_corr.select()
            else:
                self.e_use_cross_corr.deselect()

        if "ROI" in params:
            self.e_search_roi.delete(0, END)
            self.e_search_roi.insert(END, "%s" % json.dumps(params["ROI"]))


        self.l_is_computed.configure(text = "Template is precomputed.")
        #print( self.ncc.templates["t"]["pyramid"][0] )
        cv2.imshow( "Template", self.ncc.templates["t"]["pyramid"][0] )
        self.template_image = self.ncc.templates["t"]["pyramid"][0]
        self.template_mask = self.ncc.templates["t"]["mask_pyramid"][0]
        

        #lst = sorted( list(self.ncc.templates["t"]["pyramid"][0].keys()) )
        #cv2.imshow( "Template", self.ncc.templates["t"]["pyramid"][0][lst[0]] )


    
    def edit_mask( self ):
        global large_mask, large_tmpl
        large_mask = cv2.resize( self.template_mask, (0,0), fx = 2.0, fy = 2.0, interpolation=cv2.INTER_NEAREST)
        large_tmpl = cv2.resize( self.template_image, (0,0), fx = 2.0, fy = 2.0, interpolation=cv2.INTER_NEAREST)
        
        show_mask_edit()

        cv2.setMouseCallback( "Mask", mouse_handler )

        #cv2.imshow("Mask", large_mask )

    def save_mask( self ):
        global large_mask

        self.template_mask = cv2.resize( large_mask, (0,0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)
        self.l_is_computed.configure(text = "Template not computed!")

    def periodic_update( self ):
        key = cv2.waitKey(1)

        self.window.after( 100, self.periodic_update )


    def run( self ):
        self.window.mainloop()



app = AppWindow( )

app.run()
