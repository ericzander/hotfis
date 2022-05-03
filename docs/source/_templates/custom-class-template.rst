{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__init__',
                                                  '__len__',
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

   {% if '__call__' in all_methods %}
   .. automethod:: __call__
   {% endif %}

   .. automethod:: __init__

   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
