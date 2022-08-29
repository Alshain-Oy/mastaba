#!/usr/bin/env python3


import cv2
import math
import numpy as np
import sys
import time

import ctypes

import json

class Utils:
    @staticmethod
    def draw_arrow( outimg, match_x, match_y, angle ):
        cv2.line( outimg, np.int0((match_x, match_y-10)), np.int0((match_x, match_y+10)), (255,0,0), 1)
        cv2.line( outimg, np.int0((match_x-10, match_y)), np.int0((match_x+10, match_y)), (255,0,0), 1)
        
        dx = math.cos( angle * math.pi/180 )
        dy = math.sin( angle * math.pi/180 )
        
        nx = -dy
        ny = dx
    
        arrow_len = 40
        arrow_tip_size = 5
        arrow = []

        arrow.append([  match_x, match_y ])
        arrow.append([  match_x + arrow_len*dx, match_y + arrow_len*dy ])

        arrow.append([  match_x + (arrow_len - arrow_tip_size)*dx + arrow_tip_size*nx, match_y + (arrow_len - arrow_tip_size)*dy + arrow_tip_size*ny ])
        arrow.append([  match_x + (arrow_len - arrow_tip_size)*dx - arrow_tip_size*nx, match_y + (arrow_len - arrow_tip_size)*dy - arrow_tip_size*ny ])
        arrow.append([  match_x + arrow_len*dx, match_y + arrow_len*dy ])

        arrow = np.int0(arrow)
        outimg = cv2.drawContours(outimg,[arrow],0,(255,0,0),2)

        return outimg


    @staticmethod
    def find_peaks( indata, r, min_peak, max_ratio = 0.125, max_peaks = 16 ):
        peaks = []
        data = indata.copy()
        done = False
        N = 0
        while not done:
            _, max_val, _, max_loc = cv2.minMaxLoc( data )
            if max_val < min_peak:
                #print( "find_peaks, done = True, max_val < min_peak", max_val, min_peak )
                done = True
            else:
                x,y = max_loc
                if x > r and y > r:
                    area = data[ y - r : y + r + 1, x - r : x + r + 1]
                    min_area = np.median(area)
                    ratio = min_area / max_val
                    #print( "find_peaks, ratio", ratio)
                    if ratio < max_ratio:
                        peaks.append( (x, y, max_val) )
                        data[ y - r : y + r, x - r : x + r] = min_area
                    else:
                        done = True
                else:
                    data[y, x] = 0
            N += 1
            if N >= max_peaks:
                done = True
        return peaks

    @staticmethod
    def refine_peak( data, pos, from_centre = False ):
        r = 1
        x, y = pos
        area = data[ y - r : y + r + 1, x - r : x + r + 1]
        Wx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Wy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        Iarea = np.sum( area )
        Iwx = np.sum( np.multiply(area, Wx))
        Iwy = np.sum( np.multiply(area, Wy))

        if from_centre:
            return Iwx / Iarea, Iwy / Iarea
        else:    
            return Iwx / Iarea + x, Iwy / Iarea + y

    @staticmethod
    def gen_pyramid( image, levels = 3 ):
        out = {}
        out[ 0 ] = image.copy()
        img = image.copy()
        for level in range( 1, levels + 1 ):
            img = cv2.blur( img, ( 2, 2 ) )
            img = cv2.resize( img, (0,0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA )
            out[level] = img.copy()
        return out
    
    @staticmethod
    def get_padded_template_size( image ):
        imh, imw = image.shape
        padding = int( max(imh,imw) * (math.sqrt(2) - 1 ) / 2 ) 
        return imh + 2*padding, imw + 2*padding

    @staticmethod
    def synthetic_template_square( size, **kwargs ):
        padding = size // kwargs.get("borderDivisor", 4)
        plate = np.ones( (size, size), dtype = np.uint8 ) * kwargs.get("brightness", 255)
        template = cv2.copyMakeBorder(plate, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value = kwargs.get("background", 64) )

        template = cv2.blur(template, (3,3) )

        template = cv2.cvtColor( template, cv2.COLOR_GRAY2BGR )
        return template

    @staticmethod
    def synthetic_template_box( width, height, **kwargs ):
        padding = max(width, height) // kwargs.get("borderDivisor", 4)
        plate = np.ones( (height, width), dtype = np.uint8 ) * kwargs.get("brightness", 255)
        template = cv2.copyMakeBorder(plate, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value = kwargs.get("background", 64) )

        template = cv2.blur(template, (3,3) )

        template = cv2.cvtColor( template, cv2.COLOR_GRAY2BGR )
        return template

    @staticmethod
    def nms( results, nmsRadius = 100 ):
        results = sorted( results, key = lambda p: -p["score"] )
        out = []
        for entry in results:
            is_ok = True
            for tentry in out:
                d = np.linalg.norm( np.float32(entry["position"]) - np.float32(tentry["position"]) )
                if d < nmsRadius:
                    is_ok = False
            if is_ok:
                out.append( entry )
        return out

    @staticmethod
    def nmsPoints( results, nmsRadius = 100 ):
        results = sorted( results, key = lambda p: -p[2] )
        out = []
        for entry in results:
            is_ok = True
            for tentry in out:
                d = np.linalg.norm( np.float32(entry[0]) - np.float32(tentry[0]) )
                if d < nmsRadius:
                    is_ok = False
            if is_ok:
                out.append( entry )
        return out



    @staticmethod
    def compute_angle_step_auto( template, level ):
        sf = 2**(level)

        h = template.shape[0]
        w = template.shape[1]

        w /= sf
        h /= sf

        s = max([w, h])

        theta = math.atan2( 1, s )
        return theta  * 180/math.pi
    
    @staticmethod
    def read_params( fn ):
        params = {}
        with open( fn, 'r' ) as handle:
            params = json.load( handle )
        return params


class NCCUtils:
    @staticmethod
    def gen_rotations( image, angle0, angle1, dangle, **kwargs  ):
        out = {}
        masks = {}
        plain_masks = {}
        mask = np.ones(image.shape, dtype = np.uint8 ) * 255
        mask_plain = np.ones(image.shape, dtype = np.uint8 ) * 255
        mask = kwargs.get("templateMask", mask)
        
        imh, imw = image.shape
        
        
        padding = int( max(imh,imw) * (math.sqrt(2) - 1 ) / 2 ) 

        image = cv2.copyMakeBorder( image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value = 1)
        mask = cv2.copyMakeBorder( mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value = 0)
        mask_plain = cv2.copyMakeBorder( mask_plain, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value = 0)
        imh, imw = image.shape
        
        
        cx = imw // 2
        cy = imh // 2
        


        N = math.ceil( (angle1 - angle0) / dangle )

        for i in range( N ):
            q = i/(N)
            angle = angle0 * (1-q) + angle1*q
            qangle = int( angle * 100 )

            M = cv2.getRotationMatrix2D( (cx, cy), angle, 1.0  )
            out[ qangle ] = cv2.warpAffine( image, M, (imw, imh) )
            masks[ qangle ] = cv2.warpAffine( mask, M, (imw, imh) )
            plain_masks[ qangle ] = cv2.warpAffine( mask_plain, M, (imw, imh) )
        

        if kwargs.get("mustIncludeZero", False ):
            angle = 0
            qangle = 0
            M = cv2.getRotationMatrix2D( (cx, cy), angle, 1.0  )
            out[ qangle ] = cv2.warpAffine( image, M, (imw, imh) )
            masks[ qangle ] = cv2.warpAffine( mask, M, (imw, imh) )
            plain_masks[ qangle ] = cv2.warpAffine( mask_plain, M, (imw, imh) )
        

        return out, masks, plain_masks

class Precompute:
    @staticmethod
    def find_closest( angle0, angles ):
        mindelta = 1e99
        minangle = 0
        for theta in angles:
            d = abs(angle0 - theta)
            if d < mindelta:
                mindelta = d
                minangle = theta
        return mindelta, minangle

    @staticmethod
    def generate_template( img, mask = None, delta_angle = None, coarseness_factor = 1.0, Nlevels = 4, **kwargs ):
        if len(img.shape) > 2:
            bw_img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        else:
            bw_img = img.copy()
        if mask is None:
            mask = np.ones( bw_img.shape, dtype = np.uint8 ) * 255
        
        angles = {}
        pyramid = Utils.gen_pyramid( bw_img, levels = Nlevels )
        mask_pyramid = Utils.gen_pyramid( mask, levels = Nlevels )

        out = {}
        out["pyramid"] = pyramid
        out["mask_pyramid"] = mask_pyramid
        out["mastaba"] = {}
        out["angles"] = {}
        out["mapping"] = {}
        out["dtheta"] = {}
        out["params"] = {}

        for level in range( Nlevels, -1, -1):
            print( "level = %i, angleStep = %.2f deg"% (level, Utils.compute_angle_step_auto(img, level) * coarseness_factor ))
            angles[level] = []
            if delta_angle is None:
                dtheta = Utils.compute_angle_step_auto(img, level) * coarseness_factor
            else:
                dtheta = delta_angle

            #out["mastaba"][level] = NCCUtils.gen_rotations( pyramid[level], -180, 180, dtheta, templateMask = mask_pyramid[level], mustIncludeZero = True, **kwargs )
            out["mastaba"][level] = NCCUtils.gen_rotations( pyramid[level], 0, 360, dtheta, templateMask = mask_pyramid[level], mustIncludeZero = True, **kwargs )
            out["dtheta"][level] = dtheta

            angles[level] = list( out["mastaba"][level][0].keys() )
            print( "len(angles[%i]) = %i"%(level, len(angles[level])))


        angle_mapping = {}
        for level in range( Nlevels, 0, -1):
            next_level = level - 1
            angle_mapping[level] = {}
            for angle in angles[level]:
                closest = Precompute.find_closest( angle, angles[next_level] )[1]
                angle_mapping[level][angle] = closest

        out["angles"] = angles
        out["mapping"] = angle_mapping

        return out




class PyramidMatcher( object ):
    def __init__(self):
        self.templates = {}

        self.parameters = {}

    def configure( self, target, **kwargs ):
        if target not in self.parameters:
            self.parameters[target] = {}
        self.parameters[target].update( kwargs )

    def _do_matching( self, roi, tmpl, mask, **kwargs ):
        pass
    
    def _get_template_size( self, name, level ):
        pass

    def _gen_rotations( self, image, angle0, angle1, dangle, **kwargs ):
        pass

    def _get_roi( self, image, x0, y0, x1, y1 ):
        pass

    def _gen_pyramid( self, image ):
        pass

    def add_template( self, name, fn, **kwargs ):
        args = {}
        #args.update( self.parameters )
        args.update( self.parameters.get(name, {}) )
        args.update( kwargs )
        
        data = np.load( fn, allow_pickle = True )

        self.templates[name] = {}
        self.templates[name]["pyramid"] = data["pyramid"][()]
        self.templates[name]["mask_pyramid"] = data["mask_pyramid"][()]
        self.templates[name]["mastaba"] = data["mastaba"][()]
        self.templates[name]["angles"] = data["angles"][()]
        self.templates[name]["mapping"] = data["mapping"][()]
        self.templates[name]["dtheta"] = data["dtheta"][()]
        self.templates[name]["params"] = data["params"][()]
        
        
        #searchLevel = args.get("searchLevel", 3)
        angleStart = args.get("angleStart", 0)
        angleStop = args.get("angleStop", 360)
        angleStep = args.get("angleStep", 1)

        self.templates[name]["search"] = {}


        for level in range( kwargs.get("numLevels", 3) + 1 ):
            stepSize = math.floor( angleStep / self.templates[name]["dtheta"][level] )
            if stepSize < 1:
                stepSize = 1
            
            idx0 = int(  angleStart * 100 )
            idx1 = int(  angleStop * 100 )
            

            #print( "template, searchLevel:", searchLevel )
            search_area = [angle for angle in self.templates[name]["angles"][level] if angle >= idx0 and angle <= idx1 ]
            #print( "template, angles:", self.templates[name]["angles"][searchLevel] )

            self.templates[name]["search"][level] = search_area[::stepSize]

    def add_template_noload( self, name, data, **kwargs ):
        args = {}
        args.update( self.parameters.get(name, {}) )
        args.update( kwargs )
        
        self.templates[name] = {}
        self.templates[name]["pyramid"] = data["pyramid"]
        self.templates[name]["mask_pyramid"] = data["mask_pyramid"]
        self.templates[name]["mastaba"] = data["mastaba"]
        self.templates[name]["angles"] = data["angles"]
        self.templates[name]["mapping"] = data["mapping"]
        self.templates[name]["dtheta"] = data["dtheta"]
        self.templates[name]["params"] = data["params"]
        
        
        angleStart = args.get("angleStart", 0)
        angleStop = args.get("angleStop", 360)
        angleStep = args.get("angleStep", 1)

        self.templates[name]["search"] = {}
        
        for level in range( kwargs.get("numLevels", 3) + 1 ):
            stepSize = math.floor( angleStep / self.templates[name]["dtheta"][level] )
            if stepSize < 1:
                stepSize = 1
            
            idx0 = int(  angleStart * 100 )
            idx1 = int(  angleStop * 100 )
            

            #print( "template, searchLevel:", searchLevel )
            search_area = [angle for angle in self.templates[name]["angles"][level] if angle >= idx0 and angle <= idx1 ]
            #print( "template, angles:", self.templates[name]["angles"][searchLevel] )

            self.templates[name]["search"][level] = search_area[::stepSize]

    def _search( self, target, image, **kwargs ):
        args = {}
        args.update( self.parameters.get(target, {}) )
        args.update( kwargs )

        searchLevel = args.get("searchLevel", 3)
        refineLevel = args.get("refineLevel", 2)
        
        if image is not None:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            pyramid = self._gen_pyramid( image, numLevels = searchLevel, **args )
            
            self.pyramid = pyramid
        else:
            pyramid = self.pyramid    

        

        min_corr = args.get( "searchCorrelation", 0.75 )
        min_corr_verify = args.get( "searchCorrelationVerify", min_corr )

        scale = 2**searchLevel

        #print( "args", args )

        candidates = {}
   
     
        for angle in self.templates[target]["search"][searchLevel]:
            tmpl = self.templates[target]["mastaba"][searchLevel][0][angle]
            mask = self.templates[target]["mastaba"][searchLevel][1][angle]
            plain_mask = self.templates[target]["mastaba"][searchLevel][2][angle]
           
           
            res = self._do_matching( pyramid[searchLevel], tmpl, mask, plain_mask = plain_mask, min_score = min_corr, **args )
            
            if args.get("debug", False):
                #print( "search: np.max(res) =", np.max(res) )
                print( "search (level=%i, angle=%.1f): np.max(res) = %.3f, np.min(res) = %.3f"%(searchLevel, angle / 100.0, np.max(res), np.min(res)) )
                
                cv2.imshow( "pyramid[searchLevel]", pyramid[searchLevel])
                cv2.imshow( "Search: Template+mask", np.hstack([tmpl, np.uint8(mask/np.max(mask)*255)]))
                
                
                tmp = cv2.cvtColor( pyramid[searchLevel], cv2.COLOR_GRAY2BGR )
                idx = np.where( res > min_corr )
                tmp[idx] = (0,255,0)

                #cv2.imshow( "Results", tmp )


                
                
                cv2.waitKey(1)
            
            
            peaks = []
            max_val = np.max(res)
            if math.isfinite( max_val ):
                peaks = Utils.find_peaks( res, 5, min_corr, max_ratio=1.01 )
            candidates[angle] = []
            for peak in peaks:
                if peak[0] > 1 and peak[1] > 1:
                    if peak[0] < (res.shape[1] - 1) and peak[1] < (res.shape[0] - 1):
                        rx, ry = Utils.refine_peak( res, peak[:2] ) 

                        if args.get("verifyMatch", True ):
                            verified_peak = self._verify_match( pyramid[searchLevel], tmpl, mask, peak, 0, 0 )
                            if args.get( "debug", False):
                                print( "\tsearch, peak: %.3f, verified peak: %.3f" %(peak[2], verified_peak) )
                            if verified_peak > min_corr_verify:
                                candidates[angle].append( (rx * scale, ry * scale, verified_peak ) )
                        else:
                            candidates[angle].append( (rx * scale, ry * scale, peak[2] ) )
        nms = {}
        
        
        nms_candidates = []
        for angle in candidates:
            for p in candidates[angle]:
                nms_candidates.append( (p, angle, searchLevel) )

        nms_candidates = sorted( nms_candidates, key = lambda p: -p[0][2] )
        
        candidates = []
        
        
        dmin = args.get("nmsRadius", 32) / scale
        for cnd in nms_candidates:
            is_ok = True
            for cnd0 in candidates:
                dx = cnd0[0][0] - cnd[0][0]
                dy = cnd0[0][1] - cnd[0][1]
                d = dx**2 + dy**2
                if d < dmin*dmin:
                    is_ok = False
            if is_ok:
                candidates.append( cnd )

   
        results = []
        for candidate in candidates:
           
            done = False
            ref_candidate = candidate
            while not done:
                r = self._refine_search( target, pyramid, ref_candidate, **args )
                if len( r ) > 0:
                    if r[0][2] > refineLevel:
                        ref_candidate = r[0]
                    else:
                        results.extend(r)
                        done = True
                else:
                    done = True
            
        

        out = []
        for result in results:
            th, tw = self._get_template_size( target, result[2] )
            
            S = 2**result[2]
            rx = result[0][0] + (tw/2) * S
            ry = result[0][1] + (th/2) * S
            out.append( ( (rx, ry), result[1], result[0][2] ))

        return out

    def _refine_search( self, target, pyramid, candidate, **kwargs  ):
        args = {}
        args.update( self.parameters.get(target, {}) )
        args.update( kwargs )
        
        level = candidate[2] - 1
        scale = 2**level
        
        coarse_angle = self.templates[target]["mapping"][candidate[2]][ candidate[1] ]

        image = pyramid[level]
        template = self.templates[target]["mastaba"][level][0][ coarse_angle ]
        mask = self.templates[target]["mastaba"][level][1][ coarse_angle]
        plain_mask = self.templates[target]["mastaba"][level][2][ coarse_angle]
        
        if args.get( "useTemplatePadding", True):
            th, tw = Utils.get_padded_template_size( template )
        else:
            th, tw = template.shape

        divisor = args.get("refineRoiDivisor", 4)

        if False:
            x0 = int( candidate[0][0]/scale - tw//divisor )
            y0 = int( candidate[0][1]/scale - th//divisor )
            x1 = int( candidate[0][0]/scale + tw + tw//divisor )
            y1 = int( candidate[0][1]/scale + th + th//divisor )

        tpad = 5
        x0 = int( candidate[0][0]/scale - tpad )
        y0 = int( candidate[0][1]/scale - tpad )
        x1 = int( candidate[0][0]/scale + tw + tpad )
        y1 = int( candidate[0][1]/scale + th + tpad )


        if x0 < 0 or y0 < 0:
            return []
        
        if isinstance( image, tuple ):
            if x1 > image[0].shape[1] or y1 > image[0].shape[0]:
                return []
        else:
            if x1 > image.shape[1] or y1 > image.shape[0]:
                return []


        roi = self._get_roi( image, x0, y0, x1, y1 )
        
        candidates = {}
        
        min_corr = args.get( "refineCorrelation", 0.75 )

        idx1 = self.templates[target]["angles"][level].index( coarse_angle )
        idx0 = idx1 - 1
        idx2 = idx1 + 1
        idx2 = idx2 % len( self.templates[target]["angles"][level] )

        angles = [] 
        angles.append( self.templates[target]["angles"][level][idx0] )
        angles.append( self.templates[target]["angles"][level][idx1] )
        angles.append( self.templates[target]["angles"][level][idx2] )
        

        for angle in angles:
            tmpl = self.templates[target]["mastaba"][level][0][angle]
            mask = self.templates[target]["mastaba"][level][1][angle]
            plain_mask = self.templates[target]["mastaba"][level][2][angle]
            #cv2.imshow( "debug", mask )

            res = self._do_matching( roi, tmpl, mask, plain_mask = plain_mask, min_score = min_corr, **kwargs  )
            
            if args.get("debug", False):
                print( "refine (level=%i, angle=%.1f): np.max(res) = %.3f"%(level, angle/100.0, np.max(res)) )
                #cv2.imshow( "Refine: Template+mask", np.hstack([tmpl, mask*255]))
                #cv2.waitKey(100)
            

            peaks = Utils.find_peaks( res, 1, min_corr, max_ratio=1.01, max_peaks=1 )
            if args.get("debug", False):
                print( "refine (level=%i, angle=%.1f): len(peaks) = %i" % (level, angle/100.0, len(peaks)) )
                #if len(peaks) < 1:
                #    _, max_val, _, max_loc = cv2.minMaxLoc( res )
                #    print( "*****, max_val", max_val, "max_loc", max_loc, "res.shape", res.shape )

            candidates[angle] = []
            for peak in peaks:
                if peak[0] > 1 and peak[1] > 1:
                    if peak[0] < (res.shape[1] - 1) and peak[1] < (res.shape[0] - 1):
                        rx, ry = Utils.refine_peak( res, peak[:2] ) 
                        if args.get("verifyMatch", True):
                            verified_peak = self._verify_match( image, tmpl, mask, peak, x0, y0 )
                            if args.get("debug", False):
                                print( "refine (level=%i), peak: %.2f, verified peak: %.2f"%(level, peak[2], verified_peak))
                            if verified_peak > min_corr:
                                candidates[angle].append( ((rx + x0) * scale, (ry + y0) * scale, verified_peak ) )
                        else:
                            candidates[angle].append( ((rx + x0) * scale, (ry + y0) * scale, peak[2] ) )

        nms = {}
        for angle in candidates:
            for p in candidates[angle]:
                qx = int( p[0] // 32 )
                qy = int( p[1] // 32 )
                
                key = (qx, qy)
                if key not in nms:
                    nms[key] = []
                
                nms[key].append( (p, angle, level ) )
        #print( "len(nms) =", len(nms))
        candidates = []
        nms_candidates = []
        for key in nms:
            nms_candidates.extend( nms[key] )

        nms_candidates = sorted( nms_candidates, key = lambda p: -p[0][2] )
        dmin = args.get("nmsRadius", 32) / scale
        #print( "refine, len(nms_candidates) =", len(nms_candidates))
        
        for cnd in nms_candidates:
            is_ok = True
            for cnd0 in candidates:
                dx = cnd0[0][0] - cnd[0][0]
                dy = cnd0[0][1] - cnd[0][1]
                d = dx**2 + dy**2
                if d < dmin * dmin:
                    is_ok = False
            if is_ok:
                candidates.append( cnd )
                
        if args.get("debug", False):
            print( "refine (level=%i), len(candidates) =" % level, len(candidates))
        return candidates
    
    def _translate( self, image, target, point, **kwargs ):
        args = {}
        args.update( self.parameters.get( target, {}) )
        args.update( kwargs )
        #[ROI[0] + point[0][0], ROI[1] + point[0][1]]
        ROI = args.get("ROI")
        x = point[0] + ROI[0] - args.get("padding_x", 0)
        y = point[1] + ROI[1] - args.get("padding_y", 0)
        
        if args.get("fromCentre", False ):
            x -= image.shape[1] / 2
            y -= image.shape[0] / 2
        
        x *= args.get("scaleX", 1.0 )
        y *= args.get("scaleY", 1.0 )
        

        return [x, y]

        
    def _verify_match( self, image, template, mask, peak, x0, y0 ):
        qh, qw = template.shape
        nroi = self._get_roi( image, int(peak[0]+x0), int(peak[1]+y0), int(peak[0]+x0 + qw), int(peak[1]+y0 + qh) )
        masked_roi = cv2.bitwise_or( nroi, nroi, mask = mask )
        masked_tmpl = cv2.bitwise_or( template, template, mask = mask )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3) )
        eroded_mask = cv2.erode(mask, kernel, iterations = 1 )
        #res = cv2.matchTemplate( masked_roi, masked_tmpl, cv2.TM_CCOEFF_NORMED)#, mask = eroded_mask )
        res = cv2.matchTemplate( masked_roi, masked_tmpl, cv2.TM_CCOEFF_NORMED, mask = eroded_mask )
        #res2 = cv2.matchTemplate( masked_roi, masked_tmpl, cv2.TM_CCOEFF, mask = eroded_mask )
        _, maxVal, _, _ = cv2.minMaxLoc( res )
        #_, maxVal2, _, _ = cv2.minMaxLoc( res2 )
        
        #print( "_verify_match, max_val", maxVal, "maxval2", maxVal2 / np.sum(masked_tmpl) )
        #cv2.imshow("_verify_match", np.hstack([masked_roi, masked_tmpl]))
        #cv2.waitKey(1)
        
        return maxVal                                



    def detect( self, target, image, **kwargs ):
        args = {}
        args.update( self.templates[target]["params"] )
        
        args.update( self.parameters.get(target, {}) )
        args.update( kwargs )

        
        
        
        outimg = image.copy()
        
        
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.original_image = image.copy()


        ROI = args.get("ROI", (0,0, image.shape[1], image.shape[0]))
        args["ROI"] = ROI

        roi_image = image[ ROI[1] : ROI[1] + ROI[3], ROI[0] : ROI[0] + ROI[2] ]
        


        if args.get("extraPadding", False):
            th, tw = self._get_template_size( target, 0 )
            padding_y = (3*th)//2
            padding_x = (3*tw)//2

            padding_y = th//4
            padding_x = tw//4

            if args.get("useTemplatePadding", False ):
                padding_y = th//2
                padding_x = tw//2
    
            

            args["padding_y"] = padding_y
            args["padding_x"] = padding_x
            self.padding_x = padding_x
            self.padding_y = padding_y
            

            roi_image = cv2.copyMakeBorder( roi_image, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT, value = 1)
        else:
            self.padding_x = 0
            self.padding_y = 0
            

            
        
        results = self._search(target, roi_image, **args )

        template_size = max(self.templates[target]["mastaba"][0][0][0].shape)/2
        results = Utils.nmsPoints( results, template_size )

        out = []
        for point in results:
            
            px = int( point[0][0] + ROI[0] - args.get("padding_x", 0.0) )
            py = int( point[0][1] + ROI[1] - args.get("padding_y", 0.0)  )
            if args.get("drawResults", True):
                Utils.draw_arrow(outimg, px, py, -point[1] / 100.0 )
            out.append( {"angle": -point[1] / 100.0, "position": self._translate( image, target, point[0], **args ), "score": point[2] } )
        
        if args.get("scoreOrdering", True):
            out = sorted( out, key = lambda p: -p["score"] )

        if args.get("radialOrdering", False):
            out = sorted( out, key = lambda p: p["position"][0]**2 + p["position"][1]**2)
        
        if args.get("drawROI", True ):
            cv2.rectangle( outimg, (args["ROI"][0], args["ROI"][1]), (args["ROI"][0]+args["ROI"][2], args["ROI"][1]+args["ROI"][3]), (255,0,0), 1)
        


        return out, outimg


    def find_errors( self, nccimg, target, R, err_threshold ):
        tmpl = self.templates[target]["mastaba"][0][0][ int(-R["angle"]*100) ]
        mask = self.templates[target]["mastaba"][0][1][ int(-R["angle"]*100) ]
        
        
        th, tw = tmpl.shape

        x0 = int(R["position"][0] - tmpl.shape[1]/2) + self.padding_x
        y0 = int(R["position"][1] - tmpl.shape[0]/2) + self.padding_y
        x1 = x0 + tmpl.shape[1]
        y1 = y0 + tmpl.shape[0]
        #roi = self.original_image[y0:y1, x0:x1]
        print( "find_errors: roi:", x0, y0, x1, y1)
        roi = self.pyramid[0][y0:y1, x0:x1]
        
        
        original_roi = roi.copy()
        #cv2.imshow( "template", tmpl)
        #cv2.waitKey(-1)
        
        rot_excl = mask.copy()
        rot_tmpl = tmpl.copy()
        rot_excl = rot_excl[:roi.shape[0], :roi.shape[1]]
        rot_tmpl = rot_tmpl[:roi.shape[0], :roi.shape[1]]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7) )
        rot_excl = cv2.erode(rot_excl, kernel, iterations = 1 )
                            
        idx = np.where( rot_excl < 1)
        idx2 = np.where( rot_excl > 0)
        std_roi = np.std( roi )
        roi[idx] = 0
        rot_tmpl[idx] = 0

  
        delta = np.abs(np.float32( roi ) - np.float32( rot_tmpl ))
        delta[np.where(delta < err_threshold)] = 0
        
        shifted_mask_idx = (idx2[0] + y0 - self.padding_y, idx2[1] + x0 - self.padding_x)
  
        error_idx = np.where( delta > 1 )
        shifted_error_idx = (error_idx[0] + y0- self.padding_y, error_idx[1] + x0 - self.padding_x)
        nccimg[shifted_error_idx] = (0,0,255)

        

        cv2.rectangle( nccimg, (x0 - self.padding_x, y0 - self.padding_y), (x1 - self.padding_x, y1 - self.padding_y), (255,0,0), 3 )

        return delta, error_idx, (x0 - self.padding_x, y0 - self.padding_y)




class NCCMatcher( PyramidMatcher ):
    def __init__(self):
        super().__init__()

    def _do_matching( self, roi, tmpl, mask, **kwargs ):

        method = kwargs.get( "corrMethod", cv2.TM_CCORR_NORMED)
        if kwargs.get( "useMaskForMatching", True):
            return cv2.matchTemplate( roi, tmpl, method, mask = mask )
        else:
            return cv2.matchTemplate( roi, tmpl, method, mask = kwargs.get("plain_mask") )
    
    def _get_template_size( self, name, level ):
        return Utils.get_padded_template_size( self.templates[name]["pyramid"][ level ] )

    def _gen_rotations( self, image, angle0, angle1, dangle, **kwargs ):
        return NCCUtils.gen_rotations( image, angle0, angle1, dangle, **kwargs )

    def _get_roi( self, image, x0, y0, x1, y1 ):
        return image[y0:y1, x0:x1]
    
    def _gen_pyramid( self, image, **kwargs ):
        return Utils.gen_pyramid( image, levels = kwargs.get("numLevels", 3) )





class FullViewAnalysis:

    @staticmethod
    def phase_correlation( template, image, Fimage = None, from_centre = False ):


        tbw = np.zeros(image.shape, np.float32 )
        w = template.shape[1]
        h = template.shape[0]
        
        imh, imw = image.shape
        
        if from_centre:
            y0 = imh//2 - h//2
            x0 = imw//2 - w//2
            tbw[y0:y0+h, x0:x0+w] = template[:,:]
        else:
            tbw[:h, :w] = template[:,:]
        

        At = h*w
        Ai = imh*imw
        
        scaleFactor = math.sqrt(Ai / At)
        
        if Fimage is None:
            F_img = cv2.dft( image, flags = cv2.DFT_COMPLEX_OUTPUT )
        else:
            F_img = Fimage
        
        F_template = cv2.dft( tbw,  flags = cv2.DFT_COMPLEX_OUTPUT )

        mulSpec = cv2.mulSpectrums( F_img, F_template, 0 , conjB = True )

        Re, Im = cv2.split( mulSpec )

        P = np.sqrt( Re**2 + Im**2 ) + 1e-10
        


        Re = np.divide( Re, P )
        Im = np.divide( Im, P )

        mulSpec = cv2.merge( [ Re, Im ] )

        poc = cv2.idft( mulSpec, flags = cv2.DFT_REAL_OUTPUT )

        return poc / ( imh * imw ) 

    @staticmethod
    def phase_correlate_orientation( template, image, Fimage = None, from_centre = False  ):
        tbw = np.zeros(image.shape, np.float32 )
        w = template.shape[1]
        h = template.shape[0]
        
        imh, imw = image.shape
        
        if from_centre:
            y0 = imh//2 - h//2
            x0 = imw//2 - w//2
            tbw[y0:y0+h, x0:x0+w] = template[:,:]
        else:
            tbw[:h, :w] = template[:,:]
        

        At = h*w
        Ai = imh*imw
        
        scaleFactor = math.sqrt(Ai / At)
        
        

        if Fimage is None:
            F_img = cv2.dft( image - np.mean(image), flags = cv2.DFT_COMPLEX_OUTPUT )
        else:
            F_img = Fimage
        
        F_template = cv2.dft( tbw - np.mean(tbw),  flags = cv2.DFT_COMPLEX_OUTPUT )

        
        Re_F, Im_F = cv2.split( F_img )
        Re_FT, Im_FT = cv2.split( F_template )

        P_F = np.sqrt( Re_F**2 + Im_F**2 )
        P_FT = np.sqrt( Re_FT**2 + Im_FT**2 ) 
        
        P_F = np.fft.fftshift( P_F )
        P_FT = np.fft.fftshift( P_FT )
        
        P_F = P_F/np.max(P_F)
        P_FT = P_FT/np.max(P_FT)
        
        P_F = np.log(1+P_F)
        P_FT = np.log(1+P_FT)
        

        P_F = cv2.resize( P_F, (0,0), fx = 2.0, fy = 2.0 )
        P_FT = cv2.resize( P_FT, (0,0), fx = 2.0, fy = 2.0 )
        
        imh, imw = P_F.shape
        
        radius = max([imw, imh])/2  

        lp_PF = cv2.warpPolar( P_F, (0,0), (imw/2, imh/2), radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG*0 )
        lp_PFT = cv2.warpPolar( P_FT, (0,0), (imw/2, imh/2),radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG*0 )

        lp_PF = np.sqrt(lp_PF / np.max(lp_PF))
        lp_PFT = np.sqrt(lp_PFT / np.max(lp_PFT))
        

        lp_PF -= np.mean( lp_PF )
        lp_PFT -= np.mean( lp_PFT )
        

        searchSpace = np.vstack([lp_PF, lp_PF])
        res = cv2.matchTemplate(searchSpace, lp_PFT, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc( res )

        
        r0 = res[max_loc[1] - 1][0]
        r1 = res[max_loc[1] + 0][0]
        r2 = res[max_loc[1] + 1][0]
        
        rI = r0 + r1 + r2
        rP = r0*(-1) + r2*(+1)
        angle_ref = 360 - (max_loc[1] + rP/rI)/searchSpace.shape[0] * (360*2)
        return angle_ref, max_val

    @staticmethod
    def compare_images(imga, imgb):
        ta = cv2.resize( imga, (0,0), fx = 0.33, fy = 0.33 )
        tb = cv2.resize( imgb, (0,0), fx = 0.33, fy = 0.33 )
        
        tmp = np.hstack([ta, tb])

        empty = np.zeros( ta.shape, np.uint8 )


        delta = np.uint8( np.abs( np.float32(ta) - np.float32(tb)  ))
        tmp2 = cv2.merge( [ta, tb, delta] )

        cv2.imshow( "Comparison", tmp )
        cv2.imshow( "Comparison 2", tmp2 )
        #cv2.waitKey(-1)


    @staticmethod
    def find_correlation( bw_imgA, bw_imgB, border ):
        template = bw_imgB[border:-border, border:-border]


        res = FullViewAnalysis.phase_correlation( np.float32(template), np.float32(bw_imgA), from_centre=False )

        res[border-5:border+5, border-5 : border+5] = 0
        res[:,0] = 0
        res[0, :] = 0



        _,_,_,maxLoc = cv2.minMaxLoc(res)

        dx, dy = maxLoc - np.int0([border, border])

        return dx, dy, template, res

    @staticmethod
    def slow_find_correlation( bw_imgA, bw_imgB, border ):
        template = bw_imgB[border:-border, border:-border]

        res = cv2.matchTemplate(bw_imgA, template, cv2.TM_CCORR_NORMED )
        
        _,_,_,maxLoc = cv2.minMaxLoc(res)
        dx, dy = maxLoc - np.int0([border, border])
        return -dx, -dy, template, res/255


    @staticmethod
    def find_image_offset( bw_imgA, bw_imgB, _border = 250 ):
        border = _border

        dx, dy, template, res = FullViewAnalysis.find_correlation( bw_imgA, bw_imgB, border )
        failed = False
        if abs(dx) > res.shape[1]//8 or abs(dy) > res.shape[0]//8:
                failed = True
        if dy == border or dx == border and not failed:
            border //= 2
            dx, dy, template, res = FullViewAnalysis.find_correlation( bw_imgA, bw_imgB, border )
            if abs(dx) > res.shape[1]//8 or abs(dy) > res.shape[0]//8:
                failed = True

            if dy == border or dx == border and not failed:
                border //= 2
                dx, dy, template, res = FullViewAnalysis.find_correlation( bw_imgA, bw_imgB, border )
                if abs(dx) > res.shape[1]//8 or abs(dy) > res.shape[0]//8:
                    failed = True

                if dy == border or dx == border and not failed:
                    border //= 2
                    dx, dy, template, res = FullViewAnalysis.find_correlation( bw_imgA, bw_imgB, border )
                    if abs(dx) > res.shape[1]//8 or abs(dy) > res.shape[0]//8:
                        failed = True
                    

        if failed:
            border = _border // 2
            dx, dy, template, res = FullViewAnalysis.slow_find_correlation( bw_imgB, bw_imgA, border )
            
            if dx == 0 and dy == 0:
                border = _border // 4
                dx, dy, template, res = FullViewAnalysis.slow_find_correlation( bw_imgB, bw_imgA, border )
        
        return dx, dy, border

    @staticmethod
    def gen_offset_image( img, dx, dy, border ):
        padding = border+50
        tmpb = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, 0, value = 0)
        tmpb = tmpb[padding - dy: padding - dy + img.shape[0], padding - dx:padding - dx + img.shape[1]]

        return tmpb


    @staticmethod
    def find_errors( outimage, image, tmpl, mask, err_threshold ):
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9) )
        
        excl_mask = cv2.erode(mask, kernel, iterations = 1 )

        roi = image.copy()
        roi_tmpl = tmpl.copy()
        
                            
        idx = np.where( excl_mask < 1)
        idx2 = np.where( excl_mask > 0)
        roi[idx] = 0
        roi_tmpl[idx] = 0

        
        delta = np.abs(np.float32( roi ) - np.float32( roi_tmpl ))
        delta[np.where(delta < err_threshold)] = 0
        error_idx = np.where( delta > 1 )
        
        outimage[error_idx] = (0,0,255)

        return delta, error_idx
