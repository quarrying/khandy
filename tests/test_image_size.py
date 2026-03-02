import unittest
from khandy.image.misc import ImageSize


class TestImageSize(unittest.TestCase):
    
    def test_initialization_valid_values(self):
        """Test initialization with valid width and height values"""
        size = ImageSize(100, 200)
        self.assertEqual(size.width, 100)
        self.assertEqual(size.height, 200)
        
    def test_initialization_invalid_values(self):
        """Test initialization with invalid width and/or height values"""
        with self.assertRaises(ValueError):
            ImageSize(-10, 20)
            
        with self.assertRaises(ValueError):
            ImageSize(10, -20)
            
        with self.assertRaises(ValueError):
            ImageSize(0, 20)
            
        with self.assertRaises(ValueError):
            ImageSize(10, 0)
    
    def test_properties(self):
        """Test property accessors"""
        size = ImageSize(100, 200)
        
        self.assertEqual(size.cols, 100)
        self.assertEqual(size.rows, 200)
        self.assertEqual(size.area, 20000)
        self.assertAlmostEqual(size.aspect_ratio, 0.5, places=6)
        self.assertAlmostEqual(size.ar, 0.5, places=6)
        self.assertEqual(size.max_side, 200)
        self.assertEqual(size.min_side, 100)
        
    def test_as_tuple(self):
        """Test as_tuple method"""
        size = ImageSize(100, 200)
        result = size.as_tuple()
        self.assertEqual(result, (100, 200))
        self.assertIsInstance(result, tuple)
        
    def test_align_up_to(self):
        """Test align_up_to method"""
        size = ImageSize(101, 203)
        
        # Test alignment of 8
        aligned_size = size.align_up_to(8)
        self.assertEqual(aligned_size.width, 104)  # Next multiple of 8 after 101
        self.assertEqual(aligned_size.height, 208)  # Next multiple of 8 after 203
        
        # Test with return_scale
        aligned_size, x_scale, y_scale = size.align_up_to(8, return_scale=True)
        self.assertEqual(x_scale, 104/101)
        self.assertEqual(y_scale, 208/203)
        
        # Test invalid alignment
        with self.assertRaises(ValueError):
            size.align_up_to(0)
        with self.assertRaises(ValueError):
            size.align_up_to(-1)
    
    def test_scale(self):
        """Test scale method"""
        size = ImageSize(100, 200)
        
        scaled_size = size.scale(1.5, 2.0)
        self.assertEqual(scaled_size.width, 150)
        self.assertEqual(scaled_size.height, 400)
        
        # Test with return_scale
        scaled_size, x_scale, y_scale = size.scale(1.5, 2.0, return_scale=True)
        self.assertEqual(x_scale, 1.5)
        self.assertEqual(y_scale, 2.0)
    
    def test_resize_width_to(self):
        """Test resize_width_to method"""
        size = ImageSize(100, 200)
        
        resized_size = size.resize_width_to(50)
        self.assertEqual(resized_size.width, 50)
        self.assertEqual(resized_size.height, 100)  # Maintains aspect ratio
        
        # Test with return_scale
        resized_size, x_scale, y_scale = size.resize_width_to(50, return_scale=True)
        self.assertEqual(x_scale, 0.5)
        self.assertEqual(y_scale, 0.5)
    
    def test_resize_width_below(self):
        """Test resize_width_below method"""
        size = ImageSize(100, 200)
        
        # Width is already below threshold
        result = size.resize_width_below(150)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 200)
        
        # Width needs to be reduced
        result = size.resize_width_below(50)
        self.assertEqual(result.width, 50)
        self.assertEqual(result.height, 100)
    
    def test_resize_width_above(self):
        """Test resize_width_above method"""
        size = ImageSize(100, 200)
        
        # Width is already above threshold
        result = size.resize_width_above(50)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 200)
        
        # Width needs to be increased
        result = size.resize_width_above(150)
        self.assertEqual(result.width, 150)
        self.assertEqual(result.height, 300)
    
    def test_resize_height_to(self):
        """Test resize_height_to method"""
        size = ImageSize(100, 200)
        
        resized_size = size.resize_height_to(100)
        self.assertEqual(resized_size.width, 50)   # Maintains aspect ratio
        self.assertEqual(resized_size.height, 100)
        
        # Test with return_scale
        resized_size, x_scale, y_scale = size.resize_height_to(100, return_scale=True)
        self.assertEqual(x_scale, 0.5)
        self.assertEqual(y_scale, 0.5)
    
    def test_resize_height_below(self):
        """Test resize_height_below method"""
        size = ImageSize(100, 200)
        
        # Height is already below threshold
        result = size.resize_height_below(250)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 200)
        
        # Height needs to be reduced
        result = size.resize_height_below(100)
        self.assertEqual(result.width, 50)
        self.assertEqual(result.height, 100)
    
    def test_resize_height_above(self):
        """Test resize_height_above method"""
        size = ImageSize(100, 200)
        
        # Height is already above threshold
        result = size.resize_height_above(150)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 200)
        
        # Height needs to be increased
        result = size.resize_height_above(250)
        self.assertEqual(result.width, 125)
        self.assertEqual(result.height, 250)
    
    def test_resize_short_to(self):
        """Test resize_short_to method"""
        # Short side is width
        size = ImageSize(100, 200)
        result = size.resize_short_to(50)
        self.assertEqual(result.width, 50)
        self.assertEqual(result.height, 100)
        
        # Short side is height
        size = ImageSize(200, 100)
        result = size.resize_short_to(50)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 50)
    
    def test_resize_short_below(self):
        """Test resize_short_below method"""
        # Short side is already below threshold
        size = ImageSize(100, 200)
        result = size.resize_short_below(150)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 200)
        
        # Short side needs to be reduced
        size = ImageSize(100, 200)
        result = size.resize_short_below(50)
        self.assertEqual(result.width, 50)
        self.assertEqual(result.height, 100)
    
    def test_resize_short_above(self):
        """Test resize_short_above method"""
        # Short side is already above threshold
        size = ImageSize(100, 200)
        result = size.resize_short_above(50)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 200)
        
        # Short side needs to be increased
        size = ImageSize(100, 200)
        result = size.resize_short_above(150)
        self.assertEqual(result.width, 150)
        self.assertEqual(result.height, 300)
    
    def test_resize_long_to(self):
        """Test resize_long_to method"""
        # Long side is width
        size = ImageSize(200, 100)
        result = size.resize_long_to(100)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 50)
        
        # Long side is height
        size = ImageSize(100, 200)
        result = size.resize_long_to(100)
        self.assertEqual(result.width, 50)
        self.assertEqual(result.height, 100)
    
    def test_resize_long_below(self):
        """Test resize_long_below method"""
        # Long side is already below threshold
        size = ImageSize(100, 200)
        result = size.resize_long_below(250)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 200)
        
        # Long side needs to be reduced
        size = ImageSize(100, 200)
        result = size.resize_long_below(100)
        self.assertEqual(result.width, 50)
        self.assertEqual(result.height, 100)
    
    def test_resize_long_above(self):
        """Test resize_long_above method"""
        # Long side is already above threshold
        size = ImageSize(100, 200)
        result = size.resize_long_above(150)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 200)
        
        # Long side needs to be increased
        size = ImageSize(100, 200)
        result = size.resize_long_above(250)
        self.assertEqual(result.width, 125)
        self.assertEqual(result.height, 250)
    
    def test_fit_contain(self):
        """Test fit_contain method"""
        size = ImageSize(100, 200)  # Aspect ratio 1:2
        dst_size = ImageSize(200, 300)  # Aspect ratio 2:3
        
        result = size.fit_contain(dst_size)
        # Height constraint: 300/200 = 1.5, width would be 100*1.5 = 150
        # Width constraint: 200/100 = 2.0, height would be 200*2.0 = 400 (>300)
        # So height is the limiting factor
        self.assertEqual(result.width, 150)
        self.assertEqual(result.height, 300)
        
        # Test with return_scale
        result, x_scale, y_scale = size.fit_contain(dst_size, return_scale=True)
        self.assertEqual(x_scale, 1.5)
        self.assertEqual(y_scale, 1.5)
    
    def test_fit_cover(self):
        """Test fit_cover method"""
        size = ImageSize(100, 200)  # Aspect ratio 1:2
        dst_size = ImageSize(200, 300)  # Aspect ratio 2:3
        
        result = size.fit_cover(dst_size)
        # Width scaling: 200/100 = 2.0, height would be 200*2.0 = 400 (>300)
        # Height scaling: 300/200 = 1.5, width would be 100*1.5 = 150 (<200)
        # So we take the larger scale (width scaling)
        self.assertEqual(result.width, 200)
        self.assertEqual(result.height, 400)
        
        # Test with return_scale
        result, x_scale, y_scale = size.fit_cover(dst_size, return_scale=True)
        self.assertEqual(x_scale, 2.0)
        self.assertEqual(y_scale, 2.0)
    
    def test_resize_below(self):
        """Test resize_below method"""
        size = ImageSize(100, 200)
        dst_size = ImageSize(50, 50)  # Both dimensions smaller than original
        
        result = size.resize_below(dst_size)
        # Scale = min(1.0, 50/100, 50/200) = min(1.0, 0.5, 0.25) = 0.25
        self.assertEqual(result.width, 25)   # 100 * 0.25
        self.assertEqual(result.height, 50)  # 200 * 0.25
        
        # Test case where no resizing needed (scale >= 1.0)
        dst_size = ImageSize(200, 300)  # Both dimensions larger than original
        result = size.resize_below(dst_size)
        self.assertEqual(result.width, 100)  # Original size preserved
        self.assertEqual(result.height, 200)
    
    def test_resize_above(self):
        """Test resize_above method"""
        size = ImageSize(100, 200)
        dst_size = ImageSize(150, 300)  # Both dimensions larger than original
        
        result = size.resize_above(dst_size)
        # Scale = max(1.0, 150/100, 300/200) = max(1.0, 1.5, 1.5) = 1.5
        self.assertEqual(result.width, 150)  # 100 * 1.5
        self.assertEqual(result.height, 300)  # 200 * 1.5
        
        # Test case where no resizing needed (scale = 1.0)
        dst_size = ImageSize(50, 100)  # Both dimensions smaller than original
        result = size.resize_above(dst_size)
        self.assertEqual(result.width, 100)  # Original size preserved
        self.assertEqual(result.height, 200)
    
    def test_resize_to_range(self):
        """Test resize_to_range method"""
        # Test case where min_side determines the scale
        size = ImageSize(100, 200)  # min_side = 100, max_side = 200
        result = size.resize_to_range(50, 150)
        # Scale = 50/100 = 0.5, max_side after scaling = 200*0.5 = 100 <= 150
        self.assertEqual(result.width, 50)   # 100 * 0.5
        self.assertEqual(result.height, 100) # 200 * 0.5
        
        # Test case where max_side determines the scale
        size = ImageSize(100, 200)  # min_side = 100, max_side = 200
        result = size.resize_to_range(50, 180)
        # First try min_side scale: 50/100 = 0.5, max_side after scaling = 200*0.5 = 100 <= 180
        # Since it doesn't exceed max_length, we don't need to adjust
        # Actually let me fix the test - if max_side after first scale exceeds max_length,
        # then we recalculate with max_length
        size = ImageSize(100, 300)  # min_side = 100, max_side = 300
        result = size.resize_to_range(50, 180)
        # Scale = 50/100 = 0.5, max_side after scaling = 300*0.5 = 150 <= 180
        self.assertEqual(result.width, 50)   # 100 * 0.5
        self.assertEqual(result.height, 150) # 300 * 0.5
        
        # Now test the case where max_side exceeds max_length after initial scaling
        size = ImageSize(100, 300)  # min_side = 100, max_side = 300
        result = size.resize_to_range(50, 100)
        # Initial scale = 50/100 = 0.5, max_side after scaling = 300*0.5 = 150 > 100
        # So we recalculate: scale = 100/300 = 1/3
        expected_width = round(100 * (1/3))
        expected_height = round(300 * (1/3))
        self.assertEqual(result.width, expected_width)
        self.assertEqual(result.height, expected_height)


if __name__ == '__main__':
    unittest.main()
    
    