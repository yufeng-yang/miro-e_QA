# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from miro2_msg/priority_peak.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct


class priority_peak(genpy.Message):
  _md5sum = "d2924675ffe77da16f66ce7eb1cbeb6d"
  _type = "miro2_msg/priority_peak"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """#	@section COPYRIGHT
#	Copyright (C) 2023 Consequential Robotics Ltd
#	
#	@section AUTHOR
#	Consequential Robotics http://consequentialrobotics.com
#	
#	@section LICENSE
#	For a full copy of the license agreement, and a complete
#	definition of "The Software", see LICENSE in the MDK root
#	directory.
#	
#	Subject to the terms of this Agreement, Consequential
#	Robotics grants to you a limited, non-exclusive, non-
#	transferable license, without right to sub-license, to use
#	"The Software" in accordance with this Agreement and any
#	other written agreement with Consequential Robotics.
#	Consequential Robotics does not transfer the title of "The
#	Software" to you; the license granted to you is not a sale.
#	This agreement is a binding legal agreement between
#	Consequential Robotics and the purchasers or users of "The
#	Software".
#	
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
#	KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#	WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
#	OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
#	OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#	SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#	

int32 stream_index
float32[2] loc_d
float32 height
float32 size
float32 azim
float32 elev

float32 size_norm
float32 volume
float32 range

int32 actioned

"""
  __slots__ = ['stream_index','loc_d','height','size','azim','elev','size_norm','volume','range','actioned']
  _slot_types = ['int32','float32[2]','float32','float32','float32','float32','float32','float32','float32','int32']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       stream_index,loc_d,height,size,azim,elev,size_norm,volume,range,actioned

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(priority_peak, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.stream_index is None:
        self.stream_index = 0
      if self.loc_d is None:
        self.loc_d = [0.] * 2
      if self.height is None:
        self.height = 0.
      if self.size is None:
        self.size = 0.
      if self.azim is None:
        self.azim = 0.
      if self.elev is None:
        self.elev = 0.
      if self.size_norm is None:
        self.size_norm = 0.
      if self.volume is None:
        self.volume = 0.
      if self.range is None:
        self.range = 0.
      if self.actioned is None:
        self.actioned = 0
    else:
      self.stream_index = 0
      self.loc_d = [0.] * 2
      self.height = 0.
      self.size = 0.
      self.azim = 0.
      self.elev = 0.
      self.size_norm = 0.
      self.volume = 0.
      self.range = 0.
      self.actioned = 0

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self.stream_index
      buff.write(_get_struct_i().pack(_x))
      buff.write(_get_struct_2f().pack(*self.loc_d))
      _x = self
      buff.write(_get_struct_7fi().pack(_x.height, _x.size, _x.azim, _x.elev, _x.size_norm, _x.volume, _x.range, _x.actioned))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    if python3:
      if sys.version_info >= (3,0): codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      end = 0
      start = end
      end += 4
      (self.stream_index,) = _get_struct_i().unpack(str[start:end])
      start = end
      end += 8
      self.loc_d = _get_struct_2f().unpack(str[start:end])
      _x = self
      start = end
      end += 32
      (_x.height, _x.size, _x.azim, _x.elev, _x.size_norm, _x.volume, _x.range, _x.actioned,) = _get_struct_7fi().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self.stream_index
      buff.write(_get_struct_i().pack(_x))
      buff.write(self.loc_d.tostring())
      _x = self
      buff.write(_get_struct_7fi().pack(_x.height, _x.size, _x.azim, _x.elev, _x.size_norm, _x.volume, _x.range, _x.actioned))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    if python3:
      if sys.version_info >= (3,0): codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      end = 0
      start = end
      end += 4
      (self.stream_index,) = _get_struct_i().unpack(str[start:end])
      start = end
      end += 8
      self.loc_d = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=2)
      _x = self
      start = end
      end += 32
      (_x.height, _x.size, _x.azim, _x.elev, _x.size_norm, _x.volume, _x.range, _x.actioned,) = _get_struct_7fi().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_2f = None
def _get_struct_2f():
    global _struct_2f
    if _struct_2f is None:
        _struct_2f = struct.Struct("<2f")
    return _struct_2f
_struct_7fi = None
def _get_struct_7fi():
    global _struct_7fi
    if _struct_7fi is None:
        _struct_7fi = struct.Struct("<7fi")
    return _struct_7fi
_struct_i = None
def _get_struct_i():
    global _struct_i
    if _struct_i is None:
        _struct_i = struct.Struct("<i")
    return _struct_i
