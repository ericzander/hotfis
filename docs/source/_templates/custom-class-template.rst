{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members:
   :exclude-members: __init__, __weakref__

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__len__',
                                                  '__call__',
                                                  '__next__',
                                                  '__iter__',
                                                  '__getitem__',
                                                  '__setitem__',
                                                  '__delitem__',
                                                  ] %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}

   {% endif %}
   {% endblock %}
